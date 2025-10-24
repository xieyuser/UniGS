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
import os
import os.path as osp
import torch

import numpy as np

import subprocess

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

from scene import Scene
import json
import time

from PIL import Image
from gaussian_renderer import render, prefilter_voxel, GaussianModel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state, normalize
from utils.image_utils import vis_normal, vis_seg, vis_depth

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    output_ply_frame0,
):
    render_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "allrenders"
    )

    render_contri_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "contribuation"
    )

    render_rgb_gt_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_rgb"
    )

    render_rgb_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_rgb"
    )

    render_depth_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_depth"
    )

    gt_depth_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_depth"
    )

    render_normal_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_normal"
    )

    gt_normal_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_normal"
    )

    render_seg_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_seg"
    )

    gt_seg_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_seg")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_rgb_gt_path, exist_ok=True)
    os.makedirs(render_rgb_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(gt_depth_path, exist_ok=True)

    os.makedirs(render_normal_path, exist_ok=True)
    os.makedirs(gt_normal_path, exist_ok=True)
    os.makedirs(render_seg_path, exist_ok=True)
    os.makedirs(gt_seg_path, exist_ok=True)
    os.makedirs(render_contri_path, exist_ok=True)

    name_list = []
    per_view_dict = {}

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        t0 = time.time()
        torch.cuda.synchronize()
        ttt0 = time.time()

        if output_ply_frame0:
            save_gsp_path = osp.join(model_path, "frame0.ply")
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                visible_mask=None,
                save_gsp_path=save_gsp_path,
            )
            print(f"finish stored to {save_gsp_path}")
            raise
        else:
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                visible_mask=voxel_visible_mask,
            )

        torch.cuda.synchronize()

        t1 = time.time()
        t_list.append(t1 - t0)

        # continue

        rendered_rgb = (
            torch.cat([render_pkg["render"], view.original_image], dim=2)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        rendered_rgb = (rendered_rgb * 255).astype(np.uint8)

        torchvision.utils.save_image(
            render_pkg["knnmap"] * 3,
            os.path.join(render_contri_path, "{0:05d}".format(view.uid) + ".png"),
        )

        rendered_depth = np.concatenate(
            [
                vis_depth(render_pkg["depth"]),
                vis_depth(view.gt_depthany_image, is_gt=True),
            ],
            axis=1,
        )

        rendered_normal = np.concatenate(
            [
                vis_normal(render_pkg["normal"]),
                vis_normal(view.gt_normals_image[0]),
            ],
            axis=1,
        )

        rendered_seg = np.concatenate(
            [
                vis_seg(torch.argmax(render_pkg["semantic"], dim=0, keepdim=True)),
                vis_seg(view.gt_semantic_image),
            ],
            axis=1,
        )

        if save_allrenders:
            # all renders
            final_images = np.concatenate(
                [rendered_rgb, rendered_depth, rendered_normal, rendered_seg],
                axis=0,
            )
            Image.fromarray(final_images).save(
                os.path.join(render_path, "{0:05d}".format(view.uid) + ".png")
            )

            # Image.fromarray(
            #     vis_seg(torch.argmax(render_pkg["semantic"], dim=0, keepdim=True))
            # ).save(os.path.join(render_path, "{0:05d}".format(view.uid) + ".png"))

        # render images
        Image.fromarray(rendered_rgb[:, : view.image_width, :]).save(
            os.path.join(render_rgb_path, "{0:05d}".format(view.uid) + ".png")
        )

        # gt images
        gt = view.original_image[0:3, :, :]
        name_list.append("{0:05d}".format(view.uid) + ".png")
        torchvision.utils.save_image(
            gt, os.path.join(render_rgb_gt_path, "{0:05d}".format(view.uid) + ".png")
        )

        # rendered depth
        np.save(
            os.path.join(render_depth_path, "{0:05d}".format(view.uid) + ".npy"),
            render_pkg["depth"].detach().cpu().numpy(),
        )

        # gt depth
        np.save(
            os.path.join(gt_depth_path, "{0:05d}".format(view.uid) + ".npy"),
            view.gt_real_depth.detach().cpu().numpy(),
        )

        # rendered normal
        np.save(
            os.path.join(render_normal_path, "{0:05d}".format(view.uid) + ".npy"),
            render_pkg["normal"].detach().cpu().numpy(),
        )

        # gt normal
        np.save(
            os.path.join(gt_normal_path, "{0:05d}".format(view.uid) + ".npy"),
            view.gt_normals_image[0].detach().cpu().numpy(),
        )

        # rendered seg
        np.save(
            os.path.join(render_seg_path, "{0:05d}".format(view.uid) + ".npy"),
            torch.argmax(render_pkg["semantic"], dim=0, keepdim=True)
            .detach()
            .cpu()
            .numpy(),
        )

        # gt seg
        np.save(
            os.path.join(gt_seg_path, "{0:05d}".format(view.uid) + ".npy"),
            view.gt_semantic_image.detach().cpu().numpy(),
        )

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f"Test FPS: \033[1;35m{fps:.5f}\033[0m")

    with open(
        os.path.join(
            model_path, name, "ours_{}".format(iteration), "per_view_count.json"
        ),
        "w",
    ) as fp:
        json.dump(per_view_dict, fp, indent=True)

    return fps


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    output_ply: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            output_ply == False,
            output_ply == False,
        )

        time.sleep(10)

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            load_train=not skip_train,
            load_test=not skip_test,
        )

        gaussians.eval()

        count = gaussians.get_anchor_count
        print(f"GaussianCount: \033[1;35m{count}\033[0m")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        train_fps, test_fps = None, None
        if not skip_train:
            train_fps = render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                output_ply_frame0=output_ply,
            )

        if not skip_test:
            test_fps = render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                output_ply_frame0=output_ply,
            )

        tm_p = os.path.join(dataset.model_path, "tm.json")
        with open(tm_p, "w") as f:
            json.dump(
                {"train_fps": train_fps, "test_fps": test_fps, "count": count},
                f,
                indent=4,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_ply", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    save_allrenders = True

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.output_ply,
    )
