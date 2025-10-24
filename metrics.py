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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from torch.nn import functional as F

import lpips
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import numpy as np
from tabulate import tabulate


from utils.loss_utils import ssim
from utils.image_utils import psnr


def print_xys(tensor):
    print(tensor.shape, tensor.dtype, tensor.min(), tensor.mean(), tensor.max())


def compute_depth_errors(gt, pred, min_depth, max_depth):
    """Computation of error metrics between predicted and ground truth depths
    Supports PyTorch tensors and filters out invalid ground truth (gt == 0).
    """
    assert (
        gt.shape == pred.shape
    ), f"Ground truth and prediction must have the same shape {gt.shape} != {pred.shape}"

    mask = (gt > min_depth) & (gt < max_depth)

    gt = gt[mask]
    pred = pred[mask]

    if gt.numel() == 0:
        return {
            "abs_rel": float("nan"),
            "sq_rel": float("nan"),
            "rmse": float("nan"),
            "rmse_log": float("nan"),
            "a1": float("nan"),
            "a2": float("nan"),
            "a3": float("nan"),
        }

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25**2).float().mean().item()
    a3 = (thresh < 1.25**3).float().mean().item()

    rmse = torch.sqrt(((gt - pred) ** 2).mean()).item()

    rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean()).item()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()

    sq_rel = torch.mean(((gt - pred) ** 2) / gt).item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


def readNPmatrix(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    npy_files = [fname for fname in os.listdir(renders_dir) if fname.endswith(".npy")]

    for fname in tqdm(
        npy_files,
        total=len(npy_files),
        desc="Load matrix",
    ):
        render = np.load(renders_dir / fname)
        gt = np.load(gt_dir / fname)

        render_tensor = torch.from_numpy(render)
        gt_tensor = torch.from_numpy(gt)

        renders.append(render_tensor)
        gts.append(gt_tensor)
        image_names.append(fname)

    return renders, gts, image_names


def readRGBImages(renders_rgb_dir, gt_rgb_dir):
    renders = []
    gts = []
    image_names = []

    npy_files = [
        fname for fname in os.listdir(renders_rgb_dir) if fname.endswith(".png")
    ]

    for fname in tqdm(
        npy_files,
        total=len(npy_files),
        desc="Load images",
    ):
        render = Image.open(renders_rgb_dir / fname)
        gt = Image.open(gt_rgb_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def semantic_miou(render_seg, gt_seg, num_classes=15):
    pred_labels = render_seg.squeeze(1).long()
    true_labels = gt_seg.squeeze(1).long()

    present_classes = torch.unique(true_labels).cpu().numpy()

    cm = confusion_matrix(
        true_labels.cpu().numpy().flatten(),
        pred_labels.cpu().numpy().flatten(),
        labels=list(range(num_classes)),
    )

    iou_per_class = []
    valid_classes = []

    for i in range(num_classes):
        if i not in present_classes:
            continue

        valid_classes.append(i)

        if cm[i, i] == 0:
            iou_per_class.append(0)
        else:
            intersection = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - intersection
            iou_per_class.append(intersection / union if union > 0 else 0)

    if len(iou_per_class) > 0:
        miou = np.mean(iou_per_class)
    else:
        miou = 0

    return torch.tensor(miou, dtype=torch.float32)


def depth_l1(render_depth, gt_depth):
    return torch.abs(render_depth - gt_depth).mean()


def normal_cosine_similarity(render_normal, gt_normal):
    render_normal = F.normalize(render_normal, dim=1)
    gt_normal = F.normalize(gt_normal, dim=1)

    cosine_sim = (render_normal * gt_normal).sum(dim=1)
    return cosine_sim.mean()


def evaluate(model_paths, min_depth, max_depth):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method

            # gt_rgb_dir = method_dir / "gt"
            # renders_rgb_dir = method_dir / "renders"

            gt_rgb_dir = method_dir / "gt_rgb"
            renders_rgb_dir = method_dir / "renders_rgb"

            gt_depth_dir = method_dir / "gt_depth"
            # renders_depth_dir = method_dir / "depths"
            renders_depth_dir = method_dir / "renders_depth"

            gt_normal_dir = method_dir / "gt_normal"
            # renders_normal_dir = method_dir / "normals"
            renders_normal_dir = method_dir / "renders_normal"

            gt_seg_dir = method_dir / "gt_seg"
            renders_seg_dir = method_dir / "renders_seg"

            renders_rgb, gts_rgb, image_names = readRGBImages(
                renders_rgb_dir, gt_rgb_dir
            )

            renders_depth, gts_depth, image_names = readNPmatrix(
                renders_depth_dir, gt_depth_dir
            )

            renders_normal, gts_normal, image_names = readNPmatrix(
                renders_normal_dir, gt_normal_dir
            )

            renders_seg, gts_seg, image_names = readNPmatrix(
                renders_seg_dir, gt_seg_dir
            )

            ssims = []
            psnrs = []
            lpipss = []

            depth_abs_rels = []
            depth_sq_rels = []
            depth_rmses = []
            depth_rmse_logs = []
            depth_a1s = []
            depth_a2s = []
            depth_a3s = []

            normal_cossimis = []
            semantic_mious = []

            for idx in tqdm(range(len(renders_rgb)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders_rgb[idx], gts_rgb[idx]))
                psnrs.append(psnr(renders_rgb[idx], gts_rgb[idx]))
                lpipss.append(lpips_fn(renders_rgb[idx], gts_rgb[idx]).detach())

                depth_errors = compute_depth_errors(
                    renders_depth[idx], gts_depth[idx], min_depth, max_depth
                )

                depth_abs_rels.append(depth_errors["abs_rel"])
                depth_sq_rels.append(depth_errors["sq_rel"])
                depth_rmses.append(depth_errors["rmse"])
                depth_rmse_logs.append(depth_errors["rmse_log"])
                depth_a1s.append(depth_errors["a1"])
                depth_a2s.append(depth_errors["a2"])
                depth_a3s.append(depth_errors["a3"])

                normal_cossimis.append(
                    normal_cosine_similarity(renders_normal[idx], gts_normal[idx])
                )
                semantic_mious.append(semantic_miou(renders_seg[idx], gts_seg[idx]))

            metrics_means = {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "Depth Abs Rel": torch.tensor(depth_abs_rels).mean().item(),
                "Depth Sq Rel": torch.tensor(depth_sq_rels).mean().item(),
                "Depth RMSE": torch.tensor(depth_rmses).mean().item(),
                "Depth RMSE Log": torch.tensor(depth_rmse_logs).mean().item(),
                "Depth A1": torch.tensor(depth_a1s).mean().item(),
                "Depth A2": torch.tensor(depth_a2s).mean().item(),
                "Depth A3": torch.tensor(depth_a3s).mean().item(),
                "Normal Cosine": torch.tensor(normal_cossimis).mean().item(),
                "mIoU": torch.tensor(semantic_mious).mean().item(),
            }

            headers = [
                "SSIM",
                "PSNR",
                "LPIPS",
                "AbsRel",
                "SqRel",
                "RMSE",
                "RMSElog",
                "A1",
                "A2",
                "A3",
                "Normal",
                "mIoU",
            ]
            data = [
                metrics_means["SSIM"],
                metrics_means["PSNR"],
                metrics_means["LPIPS"],
                metrics_means["Depth Abs Rel"],
                metrics_means["Depth Sq Rel"],
                metrics_means["Depth RMSE"],
                metrics_means["Depth RMSE Log"],
                metrics_means["Depth A1"],
                metrics_means["Depth A2"],
                metrics_means["Depth A3"],
                metrics_means["Normal Cosine"],
                metrics_means["mIoU"],
            ]

            # ??????????
            formatted_data = [f"{x:.6f}" for x in data]

            print(tabulate([formatted_data], headers=headers, tablefmt="grid"))
            print("")

            # ??????
            full_dict[scene_dir][method].update(
                {
                    "SSIM": metrics_means["SSIM"],
                    "PSNR": metrics_means["PSNR"],
                    "LPIPS": metrics_means["LPIPS"],
                    "Depth_Abs_Rel": metrics_means["Depth Abs Rel"],
                    "Depth_Sq_Rel": metrics_means["Depth Sq Rel"],
                    "Depth_RMSE": metrics_means["Depth RMSE"],
                    "Depth_RMSE_Log": metrics_means["Depth RMSE Log"],
                    "Depth_A1": metrics_means["Depth A1"],
                    "Depth_A2": metrics_means["Depth A2"],
                    "Depth_A3": metrics_means["Depth A3"],
                    "Normal_Cosine": metrics_means["Normal Cosine"],
                    "Semantic_mIoU": metrics_means["mIoU"],
                }
            )

            # ?????????
            per_view_dict[scene_dir][method].update(
                {
                    "SSIM": {
                        name: ssim
                        for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                    },
                    "PSNR": {
                        name: psnr
                        for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                    },
                    "LPIPS": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    },
                    "Depth_Abs_Rel": {
                        name: dar
                        for dar, name in zip(
                            torch.tensor(depth_abs_rels).tolist(),
                            image_names,
                        )
                    },
                    "Depth_Sq_Rel": {
                        name: dsr
                        for dsr, name in zip(
                            torch.tensor(depth_sq_rels).tolist(),
                            image_names,
                        )
                    },
                    "Depth_RMSE": {
                        name: drmse
                        for drmse, name in zip(
                            torch.tensor(depth_rmses).tolist(),
                            image_names,
                        )
                    },
                    "Depth_RMSE_Log": {
                        name: drmselog
                        for drmselog, name in zip(
                            torch.tensor(depth_rmse_logs).tolist(),
                            image_names,
                        )
                    },
                    "Depth_A1": {
                        name: da1
                        for da1, name in zip(
                            torch.tensor(depth_a1s).tolist(), image_names
                        )
                    },
                    "Depth_A2": {
                        name: da2
                        for da2, name in zip(
                            torch.tensor(depth_a2s).tolist(), image_names
                        )
                    },
                    "Depth_A3": {
                        name: da3
                        for da3, name in zip(
                            torch.tensor(depth_a3s).tolist(), image_names
                        )
                    },
                    "Normal_Cosine": {
                        name: nc
                        for nc, name in zip(
                            torch.tensor(normal_cossimis).tolist(),
                            image_names,
                        )
                    },
                    "Semantic_mIoU": {
                        name: miou
                        for miou, name in zip(
                            torch.tensor(semantic_mious).tolist(),
                            image_names,
                        )
                    },
                }
            )

        # ???????
        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument("--min_depth", default=0.1, type=float)
    parser.add_argument("--max_depth", default=20, type=float)
    args = parser.parse_args()
    evaluate(args.model_paths, args.min_depth, args.max_depth)
