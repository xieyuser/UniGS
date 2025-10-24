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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import imageio

from utils.general_utils import (
    PILtoTorch,
    disp_to_depth_depthv2,
    depth_to_disp,
    disp_to_depth,
)

from utils.definations import *
import cv2
from PIL import Image
import einops
import numpy.linalg as npl


def depth_loader(im_p, resolution):
    image = Image.open(im_p).resize(resolution)
    depth_data_uint16_loaded = np.array(image, dtype=np.uint16)
    depth_recover = (depth_data_uint16_loaded / 65535.0).astype(np.float32)

    img = torch.from_numpy(depth_recover)
    return img.unsqueeze(0)


def depth_loader_cv2(im_p, resolution):
    img = (cv2.imread(im_p, cv2.IMREAD_ANYDEPTH).astype(np.uint16) / 1000).astype(
        np.float32
    )
    img = cv2.resize(img, resolution)
    return torch.from_numpy(img).unsqueeze(0)


def depth_loader_imageio(im_p, resolution):
    img = np.asarray(imageio.imread(im_p), dtype=np.int64)

    img = cv2.resize(
        img.astype(float),
        resolution,
        interpolation=cv2.INTER_NEAREST,
    )
    return torch.from_numpy(img).unsqueeze(0).type(torch.float32) / 6553.5


def mask_loader(img_p, resolution):
    img = Image.open(img_p).convert("L").resize(resolution, resample=Image.NEAREST)

    img_np = np.array(img)
    img_mask = img_np.copy()
    img_mask[img_mask > 149.0] = 149.0

    img_mask = np.round(img_mask / 10.0).astype(np.uint8)

    img_tensor = torch.from_numpy(img_mask)

    return img_tensor.unsqueeze(0)


class BackprojectDepth2Normal(nn.Module):
    """Layer to transform a depth image into a point cloud with multi-step normal estimation"""

    def __init__(self, batch_size, height, width, steps=[1, 2, 3]):
        super(BackprojectDepth2Normal, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.steps = steps

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False,
        )

        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False
        )

    def _xyz2normal(self, xyz, step=1):
        b, hd, wd, _ = xyz.shape

        # Vertical slices (top and bottom points)
        top_vert = xyz[:, : hd - 2 * step, step : wd - step, :]
        bottom_vert = xyz[:, 2 * step : hd, step : wd - step, :]
        bottom_to_top = top_vert - bottom_vert  # Shape: [b, hd-2*step, wd-2*step, 3]

        # Horizontal slices (left and right points)
        left_horiz = xyz[:, step : hd - step, : wd - 2 * step, :]
        right_horiz = xyz[:, step : hd - step, 2 * step : wd, :]
        left_to_right = right_horiz - left_horiz  # Shape: [b, hd-2*step, wd-2*step, 3]

        # Compute normals via cross product
        xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
        xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)

        # Pad to original size
        xyz_normal = torch.nn.functional.pad(
            xyz_normal.permute(0, 3, 1, 2),
            (step, step, step, step),  # Pad left, right, top, bottom
            mode="constant",
        ).permute(0, 2, 3, 1)

        return xyz_normal

    def forward(self, infos):
        cam_points = torch.matmul(infos["inv_K"][:, :3, :3], self.pix_coords)

        cam_points = infos["depth"].view(self.batch_size, 1, -1) * cam_points
        cam_points = einops.rearrange(
            cam_points, "b c (h w) -> b h w c", h=self.height, w=self.width
        )

        normals = []
        for step in self.steps:
            normal = self._xyz2normal(cam_points, step)
            normals.append(normal)

        if len(normals) > 1:
            reference = normals[0]
            for i in range(1, len(normals)):
                dot = torch.sum(reference * normals[i], dim=-1, keepdim=True)
                normals[i] = torch.where(dot < 0, -normals[i], normals[i])

        combined_normal = torch.mean(torch.stack(normals, dim=0), dim=0)

        view_direction = cam_points / (cam_points.norm(dim=-1, keepdim=True) + 1e-7)

        dot_product = torch.sum(combined_normal * view_direction, dim=-1, keepdim=True)

        combined_normal = torch.where(
            dot_product > 0, -combined_normal, combined_normal
        )

        b, h, w, _ = combined_normal.shape
        combined_normal_flat = combined_normal.reshape(b, h * w, 3)

        world_normals_flat = torch.bmm(
            combined_normal_flat, infos["Rcw"].transpose(1, 2)
        )

        world_normals = world_normals_flat.reshape(b, h, w, 3)

        return world_normals


def normal_to_pseudo_color(normal_map_tensor):
    # Ensure the input is a PyTorch tensor
    if not isinstance(normal_map_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    # Check the shape to be 3 x h x w
    if normal_map_tensor.shape[0] != 3:
        raise ValueError(
            f"Input tensor must have shape 3 x h x w. {normal_map_tensor.shape}"
        )

    # Detach the tensor and convert it to a numpy array
    normal_map_np = normal_map_tensor.detach().cpu().numpy()

    # Normalize the components from [-1, 1] to [0, 255]
    normal_map_np = (normal_map_np + 1) / 2 * 255

    # Clip the values to ensure they are within 0 to 255
    normal_map_np = np.clip(normal_map_np, 0, 255)

    # Convert to uint8 for image representation
    pseudo_color_image = normal_map_np.astype(np.uint8)

    # Transpose the dimensions to h x w x 3 for image representation
    pseudo_color_image = pseudo_color_image.transpose(1, 2, 0)

    return pseudo_color_image


class Camera(nn.Module):
    def __init__(
        self,
        resolution,
        colmap_id,
        R,
        T,
        Intr,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        image_path,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T

        if "/tandt/" in image_path or "/db/" in image_path:
            self.Intr = np.array(
                [
                    [Intr[0] / 2, 0, Intr[2] / 2],
                    [0, Intr[1] / 2, Intr[3] / 2],
                    [0, 0, 1],
                ]
            ).astype("f4")
        else:
            self.Intr = np.array(
                [[Intr[0], 0, Intr[2]], [0, Intr[1], Intr[3]], [0, 0, 1]]
            ).astype("f4")

        self.invIntr = npl.inv(self.Intr)

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(
                getWorld2View2(
                    R,
                    T,
                    trans,
                    scale,
                    is_replica="/replica/" in image_path or "/Replicav2/" in image_path,
                )
            )
            .transpose(0, 1)
            .cuda()
        )

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if "/replica/" not in image_path and "/Replicav2/" not in image_path:
            semantic = (
                image_path.replace("images", "semantic")
                .replace(".jpg", ".png")
                .replace(".JPG", ".png")
            )
            resized_semantic_rgb = mask_loader(semantic, resolution)

            depthany = (
                image_path.replace("images", "depths")
                .replace(".jpg", ".png")
                .replace(".JPG", ".png")
            )
            resized_depth_rgb = depth_loader(depthany, resolution)
            resized_depth_rgb = disp_to_depth_depthv2(resized_depth_rgb, 1, 80)
            resized_disp_rgb = depth_to_disp(resized_depth_rgb, 1, 50)
            resized_depth_rgb = disp_to_depth(resized_disp_rgb, 1, 80)
        else:
            if "/replica/" in image_path:
                semantic = image_path.replace("rgb", "semantic_class").replace(
                    "rgb_", "semantic_class_"
                )
                resized_semantic_rgb = mask_loader(semantic, resolution)
                depthany = image_path.replace("rgb", "depth").replace("rgb_", "depth_")

                resized_depth_rgb = depth_loader_cv2(depthany, resolution)
                resized_disp_rgb = depth_to_disp(resized_depth_rgb, 0.1, 20)
                resized_depth_rgb = disp_to_depth(resized_disp_rgb, 0.1, 20)

            else:
                semantic = image_path.replace(
                    "frames/frame", "semantic_ids/semantic_id"
                ).replace(".jpg", ".png")
                resized_semantic_rgb = mask_loader(semantic, resolution)
                depthany = image_path.replace("frames/frame", "depths/depth").replace(
                    ".jpg", ".png"
                )

                resized_depth_rgb = depth_loader_imageio(depthany, resolution)
                resized_disp_rgb = depth_to_disp(resized_depth_rgb, 0.1, 20)

        self.gt_semantic_image = resized_semantic_rgb.to(self.data_device)
        self.gt_depthany_image = resized_disp_rgb.to(self.data_device)
        self.gt_real_depth = resized_depth_rgb.to(self.data_device)

        assert (
            self.gt_depthany_image.shape[-1] == self.image_width
        ), f"{self.gt_depthany_image.shape[-1]}=={self.image_width}"

        assert (
            self.gt_depthany_image.shape[-2] == self.image_height
        ), f"{self.gt_depthany_image.shape[-2]}=={self.image_height}"

        assert (
            self.gt_semantic_image.shape[-1] == self.image_width
        ), f"{self.gt_semantic_image.shape[-1]}=={self.image_width}"

        assert (
            self.gt_semantic_image.shape[-2] == self.image_height
        ), f"{self.gt_semantic_image.shape[-2]}=={self.image_height}"

        assert (
            self.original_image.shape[-1] == self.image_width
        ), f"{self.original_image.shape[-1]}=={self.image_width}"

        assert (
            self.original_image.shape[-2] == self.image_height
        ), f"{self.original_image.shape[-2]}=={self.image_height}"

        self.depthtonormal = BackprojectDepth2Normal(
            1, self.image_height, self.image_width
        ).to(self.data_device)

        self.inv_K = torch.from_numpy(self.invIntr).unsqueeze(0).to(self.data_device)

        infos = {
            "inv_K": self.inv_K,
            "depth": resized_depth_rgb.to(self.data_device),
            "Rcw": torch.from_numpy(self.R).unsqueeze(0).float().to(self.data_device),
        }

        self.gt_normals_image = self.depthtonormal(infos).permute(0, 3, 1, 2)

        if self.uid == 0:
            print(
                {
                    "fx, fy": (self.Intr[0][0], self.Intr[1][1]),
                    "cx, cy": (self.Intr[0][2], self.Intr[1][2]),
                    "w, h": (self.image_width, self.image_height),
                }
            )


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
