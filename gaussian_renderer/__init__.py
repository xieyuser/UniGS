#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat

import math

# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )

import os
import numpy as np
from plyfile import PlyData, PlyElement
import time

from gaussian_semantic_normal_rasterization import (
    GaussianRasterizationSettings as GaussianRasterizationSettings,
    GaussianRasterizer as GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH


def construct_list_of_attributes():
    l = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        l.append("f_dc_{}".format(i))
    for i in range(15):
        l.append("f_seg_{}".format(i))
    l.append("opacity")
    l.append("contri")
    for i in range(3):
        l.append("scale_{}".format(i))
    for i in range(4):
        l.append("rot_{}".format(i))
    return l


def save_ply(xyz, color, sh_objs, contri, opacity, scaling, rot, path):
    mkdir_p(os.path.dirname(path))

    xyz = xyz.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    scaling = scaling.detach().cpu().numpy()

    sh_objs = sh_objs.detach().cpu().numpy()
    contri = contri.detach().cpu().numpy()

    rot = rot.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()

    normals = np.zeros_like(xyz)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    print(
        "save_ply: ",
        xyz.shape,
        normals.shape,
        sh_objs.shape,
        contri.shape,
        color.shape,
        opacity.shape,
        scaling.shape,
        rot.shape,
    )

    attributes = np.concatenate(
        (xyz, normals, color, sh_objs, opacity, contri, np.log(scaling), rot),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def generate_neural_gaussians(
    viewpoint_camera,
    pc: GaussianModel,
    visible_mask=None,
    is_training=False,
    save_gsp_path=None,
):
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(
            pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device
        )
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]

    grid_scaling = pc.get_scaling[visible_mask]
    grid_shobj = pc.get_shobj[visible_mask]
    grid_contri = pc.get_contri[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = (
            feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
            + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
            + feat[:, ::1, :1] * bank_weight[:, :, 2:]
        )
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]

    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = (
            torch.ones_like(
                cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
            )
            * viewpoint_camera.uid
        )
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N, k]
        contri = pc.get_contri_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
        contri = pc.get_contri_mlp(cat_local_view_wodist)  # [N, k]

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = neural_opacity > 0.0
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            shobj = pc.get_shobj_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(
                torch.cat([cat_local_view_wodist, appearance], dim=1)
            )
            shobj = pc.get_shobj_mlp(
                torch.cat([cat_local_view_wodist, appearance], dim=1)
            )
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
            shobj = pc.get_shobj_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
            shobj = pc.get_shobj_mlp(cat_local_view_wodist)

    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]
    shobj = shobj.reshape([anchor.shape[0] * pc.n_offsets, 15])  # [mask]
    contri = contri.reshape([anchor.shape[0] * pc.n_offsets, 1])  # [mask]
    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, grid_shobj, grid_contri, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, "n (c) -> (n k) (c)", k=pc.n_offsets)
    concatenated_all = torch.cat(
        [concatenated_repeated, color, shobj, contri, scale_rot, offsets], dim=-1
    )

    masked = concatenated_all[mask]

    (
        scaling_repeat,
        shobj_repeat,
        contri_repeat,
        repeat_anchor,
        color,
        shobj,
        contri,
        scale_rot,
        offsets,
    ) = masked.split([6, 15, 1, 3, 3, 15, 1, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    shobj = shobj_repeat * torch.sigmoid(shobj)
    contri = contri_repeat * torch.sigmoid(contri)

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    if save_gsp_path:
        save_ply(
            xyz.clone(),
            RGB2SH(color.clone()),
            shobj.clone(),
            contri.clone(),
            opacity.clone(),
            scaling.clone(),
            rot.clone(),
            save_gsp_path,
        )

    if is_training:
        return (
            xyz,
            color,
            shobj.unsqueeze(1),
            opacity,
            contri,
            scaling,
            rot,
            neural_opacity,
            mask,
        )
    else:
        return (
            xyz,
            color,
            shobj.unsqueeze(1),
            opacity,
            contri,
            scaling,
            rot,
        )


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    knn_search_radius_factor=0.05,
    visible_mask=None,
    retain_grad=False,
    save_gsp_path=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    # t0 = time.time()
    if is_training:
        (
            xyz,
            color,
            shobj,
            opacity,
            contris,
            scaling,
            rot,
            neural_opacity,
            mask,
        ) = generate_neural_gaussians(
            viewpoint_camera,
            pc,
            visible_mask,
            is_training=is_training,
            save_gsp_path=save_gsp_path,
        )
    else:
        xyz, color, shobj, opacity, contris, scaling, rot = generate_neural_gaussians(
            viewpoint_camera,
            pc,
            visible_mask,
            is_training=is_training,
            save_gsp_path=save_gsp_path,
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )

    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        knn_search_radius_factor=knn_search_radius_factor,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    (
        rendered_image,
        rendered_objects,
        radii,
        rendered_depth,
        rendered_normal,
        rendered_alpha,
        rendered_knn,
    ) = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        opacities=opacity,
        contris=contris,
        shs=None,
        sh_objs=shobj,
        colors_precomp=color,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    # print("forward: ", time.time() - t0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "depth": rendered_depth,
            "normal": rendered_normal,
            "semantic": rendered_objects,
            "alpha": rendered_alpha,
            "knnmap": rendered_knn,
            "num_forward": xyz.shape[0],
        }

    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": rendered_depth,
            "normal": rendered_normal,
            "semantic": rendered_objects,
            "alpha": rendered_alpha,
            "knnmap": rendered_knn,
            "num_forward": xyz.shape[0],
        }


def prefilter_voxel(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    knn_search_radius_factor=0.05,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        knn_search_radius_factor=knn_search_radius_factor,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return radii_pure > 0
