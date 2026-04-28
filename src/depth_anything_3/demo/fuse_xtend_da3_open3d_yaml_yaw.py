#!/usr/bin/env python3
"""
Fuse XTEND + DA3 RGB-D frames with Open3D.

Main improvements over the older script:
- Load camera intrinsics from ROS/DA3 YAML instead of hard-coded K.
- Prefer rectified projection_matrix intrinsics for rectified RGB/depth.
- Optionally use yaw/bearing from CSV as RGB-D odometry initialization.
- Support both rgb_rectified/ and rgb/ folders.
- Use argparse instead of editing constants inside the file.

Recommended for your current pipeline:
  rgb_rectified/ + depth_npy/ + camera_rectified_pinhole_da3.yaml
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as Rot


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Return signed difference b - a in degrees, wrapped to [-180, 180]."""
    return (b_deg - a_deg + 180.0) % 360.0 - 180.0


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def matrix_from_yaml(data: dict, key: str, shape: tuple[int, int]) -> np.ndarray:
    return np.array(data[key]["data"], dtype=np.float64).reshape(shape)


def load_intrinsics_from_yaml(yaml_path: Path, prefer_projection: bool = True):
    """
    Load camera intrinsics from either:
    - ROS camera_info-style YAML with camera_matrix/projection_matrix
    - Simple YAML with fx/fy/cx/cy/image_width/image_height

    For rectified images, use projection_matrix by default.
    For raw distorted images, use camera_matrix and undistort before this script.
    """
    data = load_yaml(yaml_path)

    width = int(data.get("image_width", data.get("width", 0)))
    height = int(data.get("image_height", data.get("height", 0)))

    if prefer_projection and "projection_matrix" in data:
        P = matrix_from_yaml(data, "projection_matrix", (3, 4))
        K = P[:3, :3].copy()
        source = "projection_matrix"
    elif "camera_matrix" in data:
        K = matrix_from_yaml(data, "camera_matrix", (3, 3))
        source = "camera_matrix"
    elif all(k in data for k in ["fx", "fy", "cx", "cy"]):
        K = np.array(
            [
                [float(data["fx"]), 0.0, float(data["cx"])],
                [0.0, float(data["fy"]), float(data["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        source = "fx/fy/cx/cy"
    else:
        raise ValueError(
            f"Could not find camera intrinsics in YAML: {yaml_path}. "
            "Expected projection_matrix, camera_matrix, or fx/fy/cx/cy."
        )

    return width, height, K, source


def make_pinhole(width: int, height: int, K: np.ndarray) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )


def make_rgbd_image(
    bgr_path: str,
    depth_path: str,
    depth_trunc_m: float,
) -> o3d.geometry.RGBDImage:
    """Convert RGB/depth files into an Open3D RGBDImage. Prefer metric .npy depth."""
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
                "This looks like a visualization image, not metric depth."
            )

    target_size = (rgb.shape[1], rgb.shape[0])
    if depth_raw.shape[:2] != rgb.shape[:2]:
        print(
            f"[warn] Resizing depth {Path(depth_path).name}: "
            f"{depth_raw.shape[1]}x{depth_raw.shape[0]} -> {target_size[0]}x{target_size[1]}"
        )
        depth_raw = cv2.resize(depth_raw, target_size, interpolation=cv2.INTER_NEAREST)

    depth_raw = depth_raw.astype(np.float32)
    depth_raw[~np.isfinite(depth_raw)] = 0.0
    depth_raw[depth_raw < 0.0] = 0.0

    valid_depth = depth_raw[depth_raw > 0.0]
    if valid_depth.size == 0:
        raise ValueError(f"No valid depth values found in: {depth_path}")

    median_depth = float(np.median(valid_depth))
    max_depth = float(np.max(valid_depth))

    # Heuristic: .npy should be meters, but keep this guard for old mm depth files.
    if median_depth > 100.0 or max_depth > 100.0:
        depth_raw = depth_raw / 1000.0

    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth_raw.astype(np.float32))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=float(depth_trunc_m),
        convert_rgb_to_intensity=False,
    )


def pair_metrics(success: bool, trans: np.ndarray, info: np.ndarray) -> dict:
    rot = Rot.from_matrix(trans[:3, :3])
    rot_deg = np.linalg.norm(rot.as_rotvec()) * 180.0 / np.pi
    t = trans[:3, 3]
    return {
        "success": bool(success),
        "rot_deg": float(rot_deg),
        "translation": t.copy(),
        "t_norm": float(np.linalg.norm(t)),
        "score": float(np.mean(np.diag(info))),
    }


def accept_pair(m: dict, min_score: float, max_rot_deg: float, max_t_norm: float) -> bool:
    if not m["success"]:
        return False
    if m["score"] < min_score:
        return False
    if m["rot_deg"] < 1e-4 and m["t_norm"] < 1e-4:
        return False
    if m["rot_deg"] > max_rot_deg:
        return False
    if m["t_norm"] > max_t_norm:
        return False
    return True


def yaw_delta_init_from_csv(
    metadata_df: pd.DataFrame,
    src_idx: int,
    tgt_idx: int,
    yaw_column: str,
    yaw_sign: float,
    yaw_axis: str,
) -> Optional[np.ndarray]:
    """
    Return Open3D odometry init from yaw/bearing delta.

    Open3D RGB-D odometry transform is source camera -> target camera.
    With rectified optical camera coordinates: x=right, y=down, z=forward.

    For a level forward-looking camera, physical yaw around world/body up usually maps
    approximately to rotation around camera negative-y. That is why the default axis
    is camera_y_up, vector [0, -1, 0]. If sign is reversed, run with --yaw-sign -1.
    """
    if metadata_df is None:
        return None
    if yaw_column not in metadata_df.columns:
        return None
    if src_idx >= len(metadata_df) or tgt_idx >= len(metadata_df):
        return None

    y0 = metadata_df.iloc[src_idx][yaw_column]
    y1 = metadata_df.iloc[tgt_idx][yaw_column]

    try:
        y0 = float(y0)
        y1 = float(y1)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(y0) or not np.isfinite(y1):
        return None

    delta_deg = angle_diff_deg(y0, y1) * float(yaw_sign)
    delta_rad = np.deg2rad(delta_deg)

    if yaw_axis == "camera_y_up":
        axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    elif yaw_axis == "camera_y_down":
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif yaw_axis == "camera_z_forward":
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif yaw_axis == "camera_x_right":
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported yaw_axis: {yaw_axis}")

    R = Rot.from_rotvec(axis * delta_rad).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def find_frame_files(recording_dir: Path, rgb_dir_name: str, depth_dir_name: str):
    rgb_dir = recording_dir / rgb_dir_name
    depth_dir = recording_dir / depth_dir_name

    rgb_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        rgb_paths.extend(glob.glob(str(rgb_dir / f"*.{ext}")))
        rgb_paths.extend(glob.glob(str(rgb_dir / f"*.{ext.upper()}")))
    rgb_paths = sorted(set(rgb_paths))

    depth_paths = sorted(glob.glob(str(depth_dir / "*.npy")))

    if not rgb_paths:
        raise RuntimeError(f"No RGB images found in: {rgb_dir}")
    if not depth_paths:
        raise RuntimeError(f"No depth .npy files found in: {depth_dir}")
    if len(rgb_paths) != len(depth_paths):
        raise RuntimeError(
            f"RGB/depth count mismatch: {len(rgb_paths)} RGB, {len(depth_paths)} depth"
        )

    return rgb_paths, depth_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse DA3 RGB-D frames with Open3D.")

    parser.add_argument("--recording-dir", required=True)
    parser.add_argument("--camera-yaml", required=True)
    parser.add_argument("--metadata-csv", default="")

    parser.add_argument("--rgb-dir", default="rgb")
    parser.add_argument("--depth-dir", default="depth_npy")
    parser.add_argument("--out-ply", default="gt_accumulated_cloud_yaml_yaw.ply")

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--step", type=int, default=2)

    parser.add_argument("--prefer-projection-matrix", action="store_true", default=True)
    parser.add_argument("--use-camera-matrix", action="store_true")

    parser.add_argument("--depth-trunc-m", type=float, default=7.0)
    parser.add_argument("--depth-min-m", type=float, default=0.3)
    parser.add_argument("--depth-max-m", type=float, default=5.0)

    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--downsample-every", type=int, default=25)

    parser.add_argument("--use-yaw-init", action="store_true")
    parser.add_argument("--yaw-column", default="bearing")
    parser.add_argument(
        "--yaw-axis",
        choices=["camera_y_up", "camera_y_down", "camera_z_forward", "camera_x_right"],
        default="camera_y_up",
    )
    parser.add_argument(
        "--yaw-sign",
        type=float,
        default=1.0,
        help="Use -1 if yaw/bearing direction is reversed.",
    )
    parser.add_argument(
        "--init-mode",
        choices=["identity", "last", "yaw", "yaw_then_last"],
        default="yaw_then_last",
        help=(
            "identity: no odometry init; last: previous accepted transform; "
            "yaw: yaw delta only; yaw_then_last: yaw delta if available, else previous transform."
        ),
    )

    parser.add_argument("--min-score", type=float, default=5e4)
    parser.add_argument("--max-rot-deg", type=float, default=10.0)
    parser.add_argument("--max-t-norm", type=float, default=0.8)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    recording_dir = Path(args.recording_dir).expanduser()
    camera_yaml = Path(args.camera_yaml).expanduser()

    prefer_projection = args.prefer_projection_matrix and not args.use_camera_matrix
    yaml_width, yaml_height, K, k_source = load_intrinsics_from_yaml(camera_yaml, prefer_projection)

    rgb_paths, depth_paths = find_frame_files(recording_dir, args.rgb_dir, args.depth_dir)

    first_bgr = cv2.imread(rgb_paths[0], cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise RuntimeError(f"Could not read first RGB image: {rgb_paths[0]}")
    image_h, image_w = first_bgr.shape[:2]

    if yaml_width > 0 and yaml_height > 0 and (yaml_width, yaml_height) != (image_w, image_h):
        print(
            f"[warn] YAML size {yaml_width}x{yaml_height} != image size {image_w}x{image_h}. "
            "Using image size for Open3D, but keeping YAML intrinsics. "
            "This is only valid if the images were not resized after calibration."
        )

    print(f"[info] recording_dir: {recording_dir}")
    print(f"[info] rgb/depth count: {len(rgb_paths)}")
    print(f"[info] image size: {image_w}x{image_h}")
    print(f"[info] camera_yaml: {camera_yaml}")
    print(f"[info] intrinsics source: {k_source}")
    print(f"[info] K:\n{K}")

    pinhole = make_pinhole(image_w, image_h, K)

    metadata_df = None
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).expanduser()
    else:
        # Common names from your capture/depth scripts.
        candidates = [recording_dir / "metadata.csv", recording_dir / "metadata_depth.csv"]
        metadata_path = next((p for p in candidates if p.exists()), None)

    if metadata_path and Path(metadata_path).exists():
        metadata_df = pd.read_csv(metadata_path)
        print(f"[info] loaded metadata: {metadata_path}")
        print(f"[info] metadata columns: {list(metadata_df.columns)}")
    else:
        print("[info] no metadata CSV loaded")

    option = o3d.pipelines.odometry.OdometryOption()
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([200, 100, 50, 20])
    option.depth_diff_max = 0.05
    option.depth_min = float(args.depth_min_m)
    option.depth_max = float(args.depth_max_m)

    accumulated_pcd = o3d.geometry.PointCloud()
    cam_to_world = np.eye(4, dtype=np.float64)
    last_transformation = np.eye(4, dtype=np.float64)

    start_idx = int(args.start_idx)
    end_idx = len(rgb_paths) - 1 if args.end_idx < 0 else min(int(args.end_idx), len(rgb_paths) - 1)
    step = max(int(args.step), 1)

    print(f"[info] fusing frame pairs: start={start_idx}, end={end_idx}, step={step}")
    print(f"[info] use_yaw_init={args.use_yaw_init}, init_mode={args.init_mode}")

    # Add the start frame at identity so the cloud is not empty if the first pair is rejected.
    first_rgbd = make_rgbd_image(rgb_paths[start_idx], depth_paths[start_idx], args.depth_trunc_m)
    first_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(first_rgbd, pinhole)
    accumulated_pcd += first_pcd

    for i in range(start_idx, end_idx, step):
        src_idx = i
        tgt_idx = i + step
        if tgt_idx >= len(rgb_paths):
            break

        yaw_init = None
        if args.use_yaw_init:
            yaw_init = yaw_delta_init_from_csv(
                metadata_df=metadata_df,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                yaw_column=args.yaw_column,
                yaw_sign=args.yaw_sign,
                yaw_axis=args.yaw_axis,
            )

        if args.init_mode == "identity":
            odo_init = np.eye(4, dtype=np.float64)
        elif args.init_mode == "last":
            odo_init = last_transformation
        elif args.init_mode == "yaw":
            odo_init = yaw_init if yaw_init is not None else np.eye(4, dtype=np.float64)
        elif args.init_mode == "yaw_then_last":
            odo_init = yaw_init if yaw_init is not None else last_transformation
        else:
            raise ValueError(f"Unknown init_mode: {args.init_mode}")

        rgbd_src = make_rgbd_image(rgb_paths[src_idx], depth_paths[src_idx], args.depth_trunc_m)
        rgbd_tgt = make_rgbd_image(rgb_paths[tgt_idx], depth_paths[tgt_idx], args.depth_trunc_m)

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_src,
            rgbd_tgt,
            pinhole,
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            option,
        )

        metrics = pair_metrics(success, trans, info)
        accepted = accept_pair(metrics, args.min_score, args.max_rot_deg, args.max_t_norm)

        print(
            f"[pair {src_idx}->{tgt_idx}] success={success} accepted={accepted} "
            f"score={metrics['score']:.1f} rot={metrics['rot_deg']:.3f}deg "
            f"t={metrics['t_norm']:.3f}m"
        )

        if yaw_init is not None:
            yaw_rot_deg = np.linalg.norm(Rot.from_matrix(yaw_init[:3, :3]).as_rotvec()) * 180.0 / np.pi
            print(f"[pair {src_idx}->{tgt_idx}] yaw init rot={yaw_rot_deg:.3f} deg")

        if accepted:
            last_transformation = trans
            # Open3D odometry returns source-camera -> target-camera.
            # To keep points in the first camera frame, accumulate the inverse.
            cam_to_world = cam_to_world @ np.linalg.inv(trans)
        else:
            last_transformation = np.eye(4, dtype=np.float64)
            print(f"[pair {src_idx}->{tgt_idx}] rejected; global pose unchanged")

        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_tgt, pinhole)
        if len(target_pcd.points) == 0:
            print(f"[warn] empty point cloud for frame {tgt_idx}")
            continue

        target_pcd.transform(cam_to_world)
        accumulated_pcd += target_pcd

        if args.downsample_every > 0 and (i - start_idx) % args.downsample_every == 0:
            accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=args.voxel_size)
            print(f"[info] accumulated points: {len(accumulated_pcd.points)}")

    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=args.voxel_size)

    out_ply = Path(args.out_ply)
    if not out_ply.is_absolute():
        out_ply = recording_dir / out_ply

    o3d.io.write_point_cloud(str(out_ply), accumulated_pcd)
    print(f"[done] saved: {out_ply}")
    print(f"[done] final points: {len(accumulated_pcd.points)}")

    if args.visualize:
        o3d.visualization.draw_geometries([accumulated_pcd])


if __name__ == "__main__":
    main()
