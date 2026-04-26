#!/usr/bin/env python3
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

recording_dir = "/home/user1/GIT/sjtu_project/recording/gazebo_capture"

def make_rgbd_image(bgr_path: str, depth_path: str) -> o3d.geometry.RGBDImage:
    """
    Convert NumPy RGB/depth arrays into an Open3D RGBDImage.

    Assumptions:
    - color: HxWx3 uint8 RGB
    - depth: HxW float32 depth in meters
    """
    bgr = cv2.imread(bgr_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr_path = Path(bgr_path)
    depth_path = Path(depth_path)
    color_o3d = o3d.io.read_image(bgr_path)
    depth_o3d = o3d.io.read_image(depth_path)
    # depth[~np.isfinite(depth)] = 0.0
    # depth[depth < 0] = 0.0
    # print(depth.shape)

    # if color.ndim != 3 or color.shape[2] != 3:
    #     raise ValueError(f"Expected RGB image with shape (H, W, 3), got {color.shape}")
    # if depth.ndim != 2:
    #     raise ValueError(f"Expected depth image with shape (H, W), got {depth.shape}")

    # color = np.ascontiguousarray(color.astype(np.uint8))
    # depth = np.ascontiguousarray(depth.astype(np.float32))

    # color_o3d = o3d.geometry.Image(color.astype(np.uint8))
    # depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        # depth_scale=1.0,
        depth_trunc=7.0,
        # convert_rgb_to_intensity=False,
    )


def get_odo_init(csv_df, src_idx, tgt_idx):
    # 1. Define the rotation from ROS (X-fwd, Z-up) to Optical (Z-fwd, Y-down)
    # This is a standard static transform for camera mounting
    R_ros_to_eye = np.array([[0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]])

    def get_world_to_camera_matrix(idx):
        row = csv_df.iloc[idx]
        t = np.array([row['tx'], row['ty'], row['tz']])
        q = [row['qx'], row['qy'], row['qz'], row['qw']]

        # World -> Drone Base
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = Rot.from_quat(q).as_matrix()
        T_world_base[:3, 3] = t

        # World -> Camera Optical (Applying the coordinate swap)
        return T_world_base @ np.linalg.inv(R_ros_to_eye)

    T_tgt_cam = get_world_to_camera_matrix(tgt_idx)
    T_src_cam = get_world_to_camera_matrix(src_idx)

    # Relative move: T_init = inv(Target) * Source
    return np.linalg.inv(T_tgt_cam) @ T_src_cam

def pair_metrics(success: bool, trans: np.ndarray, info: np.ndarray) -> dict:
    rot = Rot.from_matrix(trans[:3, :3])
    euler_deg = rot.as_euler("xyz", degrees=True)
    rot_deg = np.linalg.norm(rot.as_rotvec()) * 180.0 / np.pi
    t = trans[:3, 3]
    t_norm = np.linalg.norm(t)
    score = float(np.mean(np.diag(info)))
    return {
        "success": bool(success),
        "euler_deg": euler_deg,
        "rot_deg": float(rot_deg),
        "translation": t.copy(),
        "t_norm": float(t_norm),
        "score": score,
    }

def accept_pair(m: dict) -> bool:
    if not m["success"]:
        return False
    if m["score"] < 5e4:
        return False
    if m["rot_deg"] > 5.0:
        return False
    if m["t_norm"] > 0.6:
        return False
    return True

rgb_paths = sorted(glob.glob(os.path.join(recording_dir, "rgb", "*.jpg")))
depth_paths = sorted(glob.glob(os.path.join(recording_dir, "depth", "*.png")))

# intrinsics = np.load(os.path.join(recording_dir, "intrinsics.npy"))
# # extrinsics = np.load(os.path.join(recording_dir, "extrinsics.npy"))
# yaw = 11.35
# rx, ry, rz = np.deg2rad([0,yaw,0])
# R = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
# T = np.eye(4)
# T[:3, :3] = R
# T[:3, 3] = [0.3653,0.0063,0.0217]
# extrinsics = [np.eye(4).astype(float), T.astype(float)]
assert len(rgb_paths) == len(depth_paths)

K = np.array([[185.69090062304812, 0.0, 320.5],
[0.0, 185.69090062304812, 180.5],
[ 0.0,  0.0,  1.0]])

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.02,
    sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)
N = 0
rgbd = make_rgbd_image(rgb_paths[N], depth_paths[N])
w = 640
h = 360

pinhole = o3d.camera.PinholeCameraIntrinsic(
        w, h,
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
    )

# volume.integrate(rgbd, pinhole, np.eye(4))
last_transformation = np.identity(4)
for i, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
    if i < N:
        continue
    odo_init = last_transformation
    option = o3d.pipelines.odometry.OdometryOption()
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([200, 100, 50, 20])
    option.depth_diff_max = 0.05
    option.depth_min = 0.3
    option.depth_max = 5.0

    rgbd_src = make_rgbd_image(rgb_paths[i], depth_paths[i])
    rgbd_tgt = make_rgbd_image(rgb_paths[i+1], depth_paths[i+1])

    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_src,
        rgbd_tgt,
        pinhole,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
        option,
    )
    # print(f"info.fitness = {info.fitness}")

    if success:
        # Keep track of the movement to use as the next guess
        last_transformation = trans

    else:
        # If it fails, reset to Identity to try and find the floor again
        last_transformation = np.identity(4)

    print(f"For pair {i} + {i + 1} success is {success}")
    print("success:", success)
    print("trans:\n", trans)
    print("info:\n", info)

    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_src, pinhole)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_tgt, pinhole)

    # tutorial-style check
    source_pcd.transform(trans)

    o3d.visualization.draw_geometries([target_pcd, source_pcd])


    print(option)


    # break
    if i== len(rgb_paths)-2:
        break


# [success_color_term, trans_color_term,
#  info] = o3d.pipelines.odometry.compute_rgbd_odometry(
#     rgbds[0], rgbds[1], intrinsics_list[0], odo_init,
#     o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
# print(info)
# print(trans_color_term)
# [success_hybrid_term, trans_hybrid_term,
#  info] = o3d.pipelines.odometry.compute_rgbd_odometry(
#     rgbds[0], rgbds[1], intrinsics_list[1], odo_init,
#     o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
#
# print(info)
# print(trans_hybrid_term)


# mesh = volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()
#
# pcd = mesh.sample_points_uniformly(number_of_points=300000)
# out_ply = os.path.join(recording_dir, "gt_fused_map.ply")
# o3d.io.write_point_cloud(out_ply, pcd)
#
# print("saved:", out_ply)
# o3d.visualization.draw_geometries([pcd])