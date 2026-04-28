#!/usr/bin/env python3
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

recording_dir = "/home/user1/Documents/xtend_da3_takes/xtend_rectified_depth_takextend_da3_take_20260427_190026/"

def make_rgbd_image(bgr_path: str, depth_path: str) -> o3d.geometry.RGBDImage:
    """
    Convert RGB/depth files into an Open3D RGBDImage.

    Prefer .npy metric depth over visualization PNG depth.

    Expected depth:
    - .npy float/uint depth in meters or millimeters
    - uint16 PNG in millimeters
    """
    bgr = cv2.imread(bgr_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read RGB image: {bgr_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_path_obj = Path(depth_path)
    if depth_path_obj.suffix.lower() == ".npy":
        depth_raw = np.load(depth_path_obj)
    else:
        depth_raw = cv2.imread(str(depth_path_obj), cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        raise ValueError(f"Could not read depth image: {depth_path}")

    depth_raw = np.asarray(depth_raw)

    if depth_raw.ndim == 3:
        if depth_raw.shape[-1] == 1:
            depth_raw = depth_raw[..., 0]
        else:
            raise ValueError(
                f"Depth has 3 channels: {depth_path}. "
                "This looks like a visualization depth image, not metric depth."
            )

    target_size = (rgb.shape[1], rgb.shape[0])
    if depth_raw.shape[:2] != rgb.shape[:2]:
        depth_raw = cv2.resize(depth_raw, target_size, interpolation=cv2.INTER_NEAREST)

    depth_raw = depth_raw.astype(np.float32)
    depth_raw[~np.isfinite(depth_raw)] = 0.0
    depth_raw[depth_raw < 0.0] = 0.0

    valid_depth = depth_raw[depth_raw > 0.0]
    if valid_depth.size == 0:
        raise ValueError(f"No valid depth values found in: {depth_path}")

    median_depth = float(np.median(valid_depth))
    max_depth = float(np.max(valid_depth))

    # Heuristic:
    # - meter depth is usually around 0.1..20
    # - millimeter depth is usually hundreds/thousands
    if median_depth > 100.0 or max_depth > 100.0:
        depth_raw = depth_raw / 1000.0

    depth_raw = depth_raw.astype(np.float32)

    print(
        f"Loaded depth {os.path.basename(depth_path)}: "
        f"shape={depth_raw.shape}, "
        f"min={float(valid_depth.min())}, "
        f"median={median_depth}, "
        f"max={max_depth}"
    )

    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth_raw)

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=7.0,
        convert_rgb_to_intensity=False,
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
    if m["rot_deg"] < 1e-4 and m["t_norm"] < 1e-4:
        return False
    if m["rot_deg"] > 5.0:
        return False
    if m["t_norm"] > 0.6:
        return False
    return True

rgb_paths = sorted(glob.glob(os.path.join(recording_dir, "rgb", "*.jpg")))

# Use metric .npy depth files, not depth visualization PNGs.
depth_paths = sorted(glob.glob(os.path.join(recording_dir, "depth_npy", "*.npy")))

assert len(rgb_paths) == len(depth_paths), (
    f"RGB/depth count mismatch: {len(rgb_paths)} RGB, {len(depth_paths)} depth"
)

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

first_bgr = cv2.imread(rgb_paths[N], cv2.IMREAD_COLOR)
h, w = first_bgr.shape[:2]

pinhole = o3d.camera.PinholeCameraIntrinsic(
    w, h,
    float(K[0, 0]), float(K[1, 1]),
    float(K[0, 2]), float(K[1, 2]),
)

# volume.integrate(rgbd, pinhole, np.eye(4))
last_transformation = np.identity(4)

accumulated_pcd = o3d.geometry.PointCloud()
cam_to_world = np.identity(4)

for i in range(100, len(rgb_paths) - 1,2):
    odo_init = last_transformation
    option = o3d.pipelines.odometry.OdometryOption()
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([200, 100, 50, 20])
    option.depth_diff_max = 0.05
    option.depth_min = 0.3
    option.depth_max = 5.0

    rgbd_src = make_rgbd_image(rgb_paths[i], depth_paths[i])
    rgbd_tgt = make_rgbd_image(rgb_paths[i + 1], depth_paths[i + 1])

    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_src,
        rgbd_tgt,
        pinhole,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
        option,
    )

    metrics = pair_metrics(success, trans, info)
    accepted = accept_pair(metrics)

    print(f"For pair {i} + {i + 1} success is {success}, accepted is {accepted}")
    print("trans:\n", trans)
    print("info:\n", info)

    if accepted:
        last_transformation = trans

        # Open3D RGB-D odometry returns a transform from source camera coords
        # into target camera coords. To keep all points in frame-N/world coords,
        # accumulate camera-to-world with inverse relative motion.
        cam_to_world = cam_to_world @ np.linalg.inv(trans)
    else:
        last_transformation = np.identity(4)
        print("Rejected odometry pair; reusing previous global pose for next frame.")

    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_tgt, pinhole)
    print(f"Frame {i + 1} raw point count: {len(target_pcd.points)}")

    if len(target_pcd.points) == 0:
        print(f"Skipping frame {i + 1}: empty point cloud")
        continue

    target_pcd.transform(cam_to_world)
    accumulated_pcd += target_pcd

    if i % 25 == 0:
        accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.02)
        print(f"Accumulated {len(accumulated_pcd.points)} points so far")
    if i == 180:
        break
accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.02)

out_ply = os.path.join(recording_dir, "gt_accumulated_cloud.ply")
o3d.io.write_point_cloud(out_ply, accumulated_pcd)

print("saved:", out_ply)
o3d.visualization.draw_geometries([accumulated_pcd])

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