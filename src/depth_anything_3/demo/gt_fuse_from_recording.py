#!/usr/bin/env python3
import glob
import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

recording_dir = "/home/user1/open3d_data/extract/SampleRedwoodRGBDImages"

def make_rgbd_image(bgr_path: str, depth_path: str) -> o3d.geometry.RGBDImage:
    """
    Convert NumPy RGB/depth arrays into an Open3D RGBDImage.

    Assumptions:
    - color: HxWx3 uint8 RGB
    - depth: HxW float32 depth in meters
    """
    bgr = cv2.imread(bgr_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    print(depth.shape)

    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {color.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image with shape (H, W), got {depth.shape}")

    # color = np.ascontiguousarray(color.astype(np.uint8))
    # depth = np.ascontiguousarray(depth.astype(np.float32))

    color_o3d = o3d.geometry.Image(color.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )

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

rgb_paths = sorted(glob.glob(os.path.join(recording_dir, "color", "*.png")))
depth_paths = sorted(glob.glob(os.path.join(recording_dir, "depth", "*.npy")))

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

volume.integrate(rgbd, pinhole, np.eye(4))

prev_trans = np.eye(4)
for i, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
    if i < N:
        continue
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)

    # rgbd = make_rgbd_image(rgb_path, depth_path)
    # rgbd_dest = make_rgbd_image(rgb_paths[i+1], depth_paths[i+1])
    #
    # [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
    #     rgbd, rgbd_dest, pinhole, prev_trans,
    #     o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    # print(f"INFO: {info}\n trans: {trans}")
    # # K_i = intrinsics[i]
    # # print(f"K from camera info: {K}\n, K_i from file: {K_i}")
    # # inv_trans = np.linalg.inv(trans)
    # # trans[:3, :3] = np.linalg.inv(trans[:3, :3])
    #
    # rotation = Rot.from_matrix(trans[:3, :3])
    # print(rotation.as_euler('xyz', degrees=True))
    # print(trans[:3, 3])
    # # trans[:3, 3] = [0,0,0.26]
    # trans[:3, :3] = np.linalg.inv(trans[:3, :3])
    # # trans = np.linalg.inv(trans)
    # prev_trans = trans
    #
    # volume.integrate(rgbd_dest, pinhole, trans)

    rgbd_src = make_rgbd_image(rgb_paths[i], depth_paths[i])
    rgbd_tgt = make_rgbd_image(rgb_paths[i+1], depth_paths[i+1])

    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_src,
        rgbd_tgt,
        pinhole,
        np.eye(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

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