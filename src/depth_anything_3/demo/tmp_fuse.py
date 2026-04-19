#!/usr/bin/env python3
import glob
import os
import cv2
import numpy as np
import open3d as o3d

recording_dir = "/home/user1/GIT/sjtu_project/recording"

rgb_paths = sorted(glob.glob(os.path.join(recording_dir, "rgb", "*.png")))
depth_paths = sorted(glob.glob(os.path.join(recording_dir, "depth", "*.npy")))

K = np.array([
    [185.69090062304812, 0.0, 320.5],
    [0.0, 185.69090062304812, 180.5],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

pinhole = o3d.camera.PinholeCameraIntrinsic(
    640, 360,
    float(K[0, 0]), float(K[1, 1]),
    float(K[0, 2]), float(K[1, 2]),
)

def load_rgbd(rgb_path, depth_path):
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[(depth < 0.2) | (depth > 4.0)] = 0.0

    color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb.astype(np.uint8)))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=4.0,
        convert_rgb_to_intensity=False,
    )
    return rgbd

def rgbd_to_pcd(rgbd):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    pcd = pcd.voxel_down_sample(0.03)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd

def icp_multiscale(source, target, init=np.eye(4)):
    current = init.copy()
    for voxel, dist in [(0.05, 0.20), (0.03, 0.12), (0.015, 0.06)]:
        src = source.voxel_down_sample(voxel)
        tgt = target.voxel_down_sample(voxel)

        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)
        )
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)
        )

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_correspondence_distance=dist,
            init=current,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        current = reg.transformation
    return current

rgbd0 = load_rgbd(rgb_paths[0], depth_paths[0])
pcd0 = rgbd_to_pcd(rgbd0)

global_poses = [np.eye(4)]
pcds = [pcd0]

for i in range(1, len(rgb_paths)):
    rgbd_i = load_rgbd(rgb_paths[i], depth_paths[i])
    pcd_i = rgbd_to_pcd(rgbd_i)

    T_prev_to_i = icp_multiscale(pcd_i, pcds[-1], init=np.eye(4))
    T_world_i = global_poses[-1] @ T_prev_to_i

    global_poses.append(T_world_i)
    pcds.append(pcd_i)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.02,
    sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

for i in range(len(rgb_paths)):
    rgbd = load_rgbd(rgb_paths[i], depth_paths[i])
    volume.integrate(rgbd, pinhole, global_poses[i])

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=300000)

out_ply = os.path.join(recording_dir, "gt_fused_map_icp.ply")
o3d.io.write_point_cloud(out_ply, pcd)
print("saved:", out_ply)

o3d.visualization.draw_geometries([pcd])