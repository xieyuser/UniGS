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
import cv2
import numpy as np
import matplotlib

from .definations import seg_class_colors
from .general_utils import depth_to_disp


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def vis_depth(depth_tensor, is_gt=False):
    if not is_gt:
        depth_tensor = depth_to_disp(depth_tensor, 1, 50)

    depth_tensor = depth_tensor[0].detach().cpu().numpy()

    colormap_func = matplotlib.colormaps.get_cmap("viridis").reversed()
    colored_image = colormap_func(depth_tensor)[:, :, :3]
    re = (colored_image * 255).astype("uint8")
    return re


def vis_normal(normal_map_tensor):
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

    normal_map_np = np.nan_to_num(normal_map_np, nan=0.0, posinf=1.0, neginf=0.0)
    # Convert to uint8 for image representation
    pseudo_color_image = normal_map_np.astype(np.uint8)

    # Transpose the dimensions to h x w x 3 for image representation
    pseudo_color_image = pseudo_color_image.transpose(1, 2, 0)

    return pseudo_color_image


def process_input(img):
    if torch.is_tensor(img):

        img = img.detach().cpu().numpy()

        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)

            img = img.squeeze()

    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img.astype(np.uint8)


def vis_seg(input_data, plot_name=False):

    img = process_input(input_data)

    class_names = list(seg_class_colors.keys())
    colors = np.array([seg_class_colors[name] for name in class_names], dtype=np.uint8)

    img[img == len(seg_class_colors)] = 0
    colored_img_bgr = colors[img]

    return colored_img_bgr
