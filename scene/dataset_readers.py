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

import sys

import glob
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import numpy.linalg as npl

import json
from pathlib import Path
from plyfile import PlyData, PlyElement

try:
    import laspy
except:
    print("No laspy")
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    Intr: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info, is_replica=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T, is_replica=is_replica)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # if idx >= 120:
        #     continue
        # if idx % 5 != 0:
        #     continue
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            Intr = np.array(
                [intr.params[0], intr.params[0], intr.params[2], intr.params[3]]
            )
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            Intr = np.array(
                [intr.params[0], intr.params[1], intr.params[2], intr.params[3]]
            )
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = osp.join(images_folder, osp.basename(extr.name))
        image_name = osp.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            Intr=Intr,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def readReplicaCameras(images_folder, trajectory_file="../traj_w_c.txt"):
    with open(osp.join(images_folder, trajectory_file), "r") as f:
        lines = f.readlines()

    c2w_matrices = []
    for line in lines[:]:
        data = line.strip().split()

        if len(data) == 0:
            continue

        matrix = np.array(list(map(float, data))).reshape(4, 4)
        c2w_matrices.append(matrix)

    cam_infos = []
    num_cameras = len(c2w_matrices)
    for idx in range(num_cameras):
        # if idx % 5 != 0:
        #     continue
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, num_cameras))
        sys.stdout.flush()

        c2w = c2w_matrices[idx]

        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]

        intr = np.array([320, 320, 319.5, 239.5])
        height, width = 480, 640

        FovX = focal2fov(intr[0], width)
        FovY = focal2fov(intr[1], height)

        image_path = osp.join(images_folder, f"rgb_{idx}.png")
        image_name = int(osp.basename(image_path).split(".")[0].split("_")[1])

        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=idx,
            R=R_c2w,
            T=t_c2w,
            Intr=intr,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def readReplicav2Cameras(images_folder, trajectory_file="../traj.txt"):
    with open(osp.join(images_folder, trajectory_file), "r") as f:
        lines = f.readlines()

    c2w_matrices = []

    for line in lines[:]:
        data = line.strip().split()

        if len(data) == 0:
            continue

        matrix = np.array(list(map(float, data))).reshape(4, 4)
        c2w_matrices.append(matrix)

    cam_infos = []
    num_cameras = len(c2w_matrices)
    for idx in range(num_cameras):
        if idx % 2 != 0:
            continue
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, num_cameras))
        sys.stdout.flush()

        c2w = c2w_matrices[idx]

        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]

        intr = np.array([600, 600, 599.5, 339.5])
        height, width = 680, 1200

        FovX = focal2fov(intr[0], width)
        FovY = focal2fov(intr[1], height)

        image_path = osp.join(images_folder, f"frame{str(idx).rjust(6, '0')}.jpg")
        image_name = str(
            int(osp.basename(image_path).split(".")[0].replace("frame", ""))
        ).rjust(6, "0")

        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=idx,
            R=R_c2w,
            T=t_c2w,
            Intr=intr,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path, step=1):
    step = max(1, int(step))

    plydata = PlyData.read(path)
    vertices = plydata["vertex"]

    all_positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    all_normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T

    try:
        all_colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
    except:
        all_colors = None

    indices = np.arange(0, len(vertices), step)

    positions = all_positions[indices]
    normals = all_normals[indices]

    if all_colors is not None:
        colors = all_colors[indices]
    else:
        colors = np.random.rand(len(positions), 3)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readReplicaSceneInfo(path, images, eval, lod, llffhold=8):
    reading_dir = "rgb" if images == None else images
    cam_infos_unsorted = readReplicaCameras(
        images_folder=osp.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        if lod > 0:
            print(f"using lod, using eval")
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f"test_cam_infos: {len(test_cam_infos)}")
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            test_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold == 0
            ]

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "points3D.ply")
    if not osp.exists(ply_path):
        print(
            "Converting mesh.ply to .ply, will happen only the first time you open the scene."
        )
        mesh_path = osp.join(path, "mesh.ply")
        pcd = fetchPly(mesh_path, step=5)
        storePly(ply_path, pcd.points, pcd.colors)

    print(f"start fetching data from ply file")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readReplicaV2SceneInfo(path, images, eval, lod, llffhold=8):
    reading_dir = "frames" if images == None else images
    cam_infos_unsorted = readReplicav2Cameras(
        images_folder=osp.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        if lod > 0:
            print(f"using lod, using eval")
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f"test_cam_infos: {len(test_cam_infos)}")
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            test_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold == 0
            ]

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "points3D.ply")
    if not osp.exists(ply_path):
        print(
            "Converting mesh.ply to .ply, will happen only the first time you open the scene."
        )
        mesh_path = osp.join(path, f"{osp.basename(path)}_mesh.ply")
        pcd = fetchPly(mesh_path, step=5)
        storePly(ply_path, pcd.points, pcd.colors)

    print(f"start fetching data from ply file")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readColmapSceneInfo(path, images, eval, lod, llffhold=8):
    try:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=osp.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        if lod > 0:
            print(f"using lod, using eval")
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f"test_cam_infos: {len(test_cam_infos)}")
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            test_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold == 0
            ]

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "sparse/0/points3D.ply")
    bin_path = osp.join(path, "sparse/0/points3D.bin")
    txt_path = osp.join(path, "sparse/0/points3D.txt")
    if not osp.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    # try:
    print(f"start fetching data from ply file")
    pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(
    path,
    transformsfile,
    white_background,
    extension=".png",
    is_debug=False,
    undistorted=False,
):
    cam_infos = []
    with open(osp.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split(".")[-1] in ["jpg", "jpeg", "JPG", "png"]:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])

        Ts = c2ws[:, :3, 3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = osp.join(path, frame["file_path"] + extension)
            if not osp.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            if idx % 10 == 0:
                progress_bar.set_postfix(
                    {"num": Fore.YELLOW + f"{ct}/{len(frames)}" + Style.RESET_ALL}
                )
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()

            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1, -1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = osp.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array(
                    [frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]],
                    dtype=np.float32,
                )
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                    1 - norm_data[:, :, 3:4]
                )
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

            if is_debug and idx > 50:
                break
    return cam_infos


def readNerfSyntheticInfo(
    path, white_background, eval, extension=".png", ply_path=None
):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = osp.join(path, "points3d.ply")
    if not osp.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Replica": readReplicaSceneInfo,
    "Replicav2": readReplicaV2SceneInfo,
    "Blender": readNerfSyntheticInfo,
}
