#!/usr/bin/env python3
"""
Fuse XTEND + DA3 RGB-D frames with Open3D, export trajectory, and estimate velocity/distance.

Main ideas:
- Load camera intrinsics from YAML instead of hard-coded K.
- Optionally use yaw/bearing from metadata as RGB-D odometry initialization.
- Accumulate camera pose from Open3D RGB-D odometry.
- Keep raw accepted odometry for point-cloud fusion.
- Estimate instantaneous velocity from accumulated world-frame pose deltas.
- Smooth velocity with an exponential moving average.
- Integrate smoothed speed over time for a less jitter-sensitive path-length estimate.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as Rot


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def matrix_from_yaml(data: dict, key: str, shape: tuple[int, int]) -> np.ndarray:
    return np.array(data[key]["data"], dtype=np.float64).reshape(shape)


def load_intrinsics_from_yaml(yaml_path: Path, prefer_projection: bool = True):
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
        raise ValueError(f"Could not find camera intrinsics in YAML: {yaml_path}")

    return width, height, K, source


def scale_intrinsics_if_needed(K: np.ndarray, yaml_w: int, yaml_h: int, image_w: int, image_h: int) -> np.ndarray:
    K_scaled = K.copy()
    if yaml_w > 0 and yaml_h > 0 and (yaml_w, yaml_h) != (image_w, image_h):
        sx = float(image_w) / float(yaml_w)
        sy = float(image_h) / float(yaml_h)
        K_scaled[0, 0] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx
        K_scaled[1, 2] *= sy
    return K_scaled


def make_pinhole(width: int, height: int, K: np.ndarray) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        int(width), int(height), float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    )


def make_rgbd_image(bgr_path: str, depth_path: str, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
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
            raise ValueError(f"Depth has 3 channels: {depth_path}. This is probably visualization, not metric depth.")

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

    if float(np.median(valid_depth)) > 100.0 or float(np.max(valid_depth)) > 100.0:
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

def load_depth_m(depth_path: str) -> np.ndarray:
    """Load metric depth from .npy or depth image and return float32 meters."""
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

    depth_m = depth_raw.astype(np.float32)
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0

    valid = depth_m[depth_m > 0.0]
    if valid.size > 0:
        median_depth = float(np.median(valid))
        max_depth = float(np.max(valid))

        # Guard for old millimeter depth files.
        if median_depth > 100.0 or max_depth > 100.0:
            depth_m = depth_m / 1000.0

    return depth_m.astype(np.float32)


def compute_center_depth_roi(
    depth_m: np.ndarray,
    rgb_shape: tuple[int, int],
    roi_size_px: int,
    roi_center_y_frac: float,
    min_depth_m: float,
    max_depth_m: float,
) -> dict:
    """
    Compute average depth in a square ROI centered horizontally and slightly below image middle.

    rgb_shape:
      (height, width)

    roi_center_y_frac:
      0.50 = image vertical center
      0.58 = slightly below center
    """
    rgb_h, rgb_w = rgb_shape

    if depth_m.shape[:2] != (rgb_h, rgb_w):
        depth_resized = cv2.resize(
            depth_m,
            (rgb_w, rgb_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        depth_resized = depth_m

    size = max(1, int(roi_size_px))
    cx = rgb_w // 2
    cy = int(round(float(roi_center_y_frac) * float(rgb_h)))

    half = size // 2

    x1 = max(0, cx - half)
    x2 = min(rgb_w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(rgb_h, cy + half)

    patch = depth_resized[y1:y2, x1:x2]

    valid = (
        np.isfinite(patch)
        & (patch > 0.0)
        & (patch >= float(min_depth_m))
        & (patch <= float(max_depth_m))
    )

    vals = patch[valid]

    if vals.size == 0:
        mean_m = float("nan")
        median_m = float("nan")
        std_m = float("nan")
        min_m = float("nan")
        max_m = float("nan")
    else:
        mean_m = float(np.mean(vals))
        median_m = float(np.median(vals))
        std_m = float(np.std(vals))
        min_m = float(np.min(vals))
        max_m = float(np.max(vals))

    return {
        "roi_x1": int(x1),
        "roi_y1": int(y1),
        "roi_x2": int(x2),
        "roi_y2": int(y2),
        "roi_valid_count": int(vals.size),
        "roi_total_count": int(patch.size),
        "roi_mean_depth_m": mean_m,
        "roi_median_depth_m": median_m,
        "roi_std_depth_m": std_m,
        "roi_min_depth_m": min_m,
        "roi_max_depth_m": max_m,
    }


def save_center_depth_debug_image(
    bgr_path: str,
    out_path: Path,
    roi_stats: dict,
    label: str,
) -> None:
    """Save RGB image with center-depth ROI rectangle drawn."""
    bgr = cv2.imread(bgr_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return

    x1 = int(roi_stats["roi_x1"])
    y1 = int(roi_stats["roi_y1"])
    x2 = int(roi_stats["roi_x2"])
    y2 = int(roi_stats["roi_y2"])

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    text = (
        f"{label} mean={roi_stats['roi_mean_depth_m']:.2f}m "
        f"valid={roi_stats['roi_valid_count']}/{roi_stats['roi_total_count']}"
    )

    cv2.putText(
        bgr,
        text,
        (max(5, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)

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
        raise RuntimeError(f"RGB/depth count mismatch: {len(rgb_paths)} RGB, {len(depth_paths)} depth")
    return rgb_paths, depth_paths


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return (b_deg - a_deg + 180.0) % 360.0 - 180.0


def yaw_delta_init_from_csv(
    metadata_df: Optional[pd.DataFrame],
    src_idx: int,
    tgt_idx: int,
    yaw_column: str,
    yaw_sign: float,
    yaw_axis: str,
) -> Optional[np.ndarray]:
    if metadata_df is None or yaw_column not in metadata_df.columns:
        return None
    if src_idx >= len(metadata_df) or tgt_idx >= len(metadata_df):
        return None

    try:
        y0 = float(metadata_df.iloc[src_idx][yaw_column])
        y1 = float(metadata_df.iloc[tgt_idx][yaw_column])
    except (TypeError, ValueError):
        return None

    if not np.isfinite(y0) or not np.isfinite(y1):
        return None

    delta_rad = np.deg2rad(angle_diff_deg(y0, y1) * float(yaw_sign))

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

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_rotvec(axis * delta_rad).as_matrix()
    return T


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


def get_frame_time_sec(metadata_df: Optional[pd.DataFrame], idx: int, fallback_fps: float) -> float:
    if metadata_df is not None and "stamp_sec" in metadata_df.columns and idx < len(metadata_df):
        try:
            t = float(metadata_df.iloc[idx]["stamp_sec"])
            if np.isfinite(t):
                return t
        except (TypeError, ValueError):
            pass
    return float(idx) / float(fallback_fps)


def compute_dt_sec(metadata_df: Optional[pd.DataFrame], src_idx: int, tgt_idx: int, fallback_fps: float) -> float:
    t_src = get_frame_time_sec(metadata_df, src_idx, fallback_fps)
    t_tgt = get_frame_time_sec(metadata_df, tgt_idx, fallback_fps)
    dt = t_tgt - t_src
    if not np.isfinite(dt) or dt <= 1e-6:
        dt = float(tgt_idx - src_idx) / float(fallback_fps)
    return max(float(dt), 1e-6)


def update_velocity_ema(prev_v: Optional[np.ndarray], v_inst: np.ndarray, dt: float, tau_sec: float) -> np.ndarray:
    if prev_v is None:
        return v_inst.astype(np.float64)
    alpha = 1.0 - np.exp(-dt / max(float(tau_sec), 1e-6))
    return (1.0 - alpha) * prev_v + alpha * v_inst


def transform_to_pose_fields(T: np.ndarray) -> dict:
    rot = Rot.from_matrix(T[:3, :3])
    euler_xyz_deg = rot.as_euler("xyz", degrees=True)
    return {
        "cam_x_right_m": float(T[0, 3]),
        "cam_y_down_m": float(T[1, 3]),
        "cam_z_forward_m": float(T[2, 3]),
        "roll_x_deg": float(euler_xyz_deg[0]),
        "pitch_y_deg": float(euler_xyz_deg[1]),
        "yaw_z_deg": float(euler_xyz_deg[2]),
    }


def metadata_value(metadata_df: Optional[pd.DataFrame], idx: int, column: str):
    if metadata_df is None or not column or column not in metadata_df.columns or idx >= len(metadata_df):
        return np.nan
    return metadata_df.iloc[idx][column]


def make_trajectory_line_set(positions: list[np.ndarray], color=(1.0, 0.0, 0.0)) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    if len(positions) == 0:
        return line_set
    pts = np.asarray(positions, dtype=np.float64)
    line_set.points = o3d.utility.Vector3dVector(pts)
    if len(positions) >= 2:
        lines = np.array([[i, i + 1] for i in range(len(positions) - 1)], dtype=np.int32)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(lines), 1)))
    return line_set


def make_trajectory_point_cloud(positions: list[np.ndarray], line_step_m: float = 0.02) -> o3d.geometry.PointCloud:
    traj_pcd = o3d.geometry.PointCloud()
    if len(positions) == 0:
        return traj_pcd
    sampled, colors = [], []
    for a, b in zip(positions[:-1], positions[1:]):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        dist = float(np.linalg.norm(b - a))
        n = max(int(np.ceil(dist / max(line_step_m, 1e-6))), 1)
        for j in range(n):
            t = j / float(n)
            sampled.append((1.0 - t) * a + t * b)
            colors.append([1.0, 0.0, 0.0])

    def add_marker(center, color, radius=0.05, samples=60):
        rng = np.random.default_rng(12345)
        for _ in range(samples):
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            sampled.append(np.asarray(center, dtype=np.float64) + radius * direction / norm)
            colors.append(color)

    add_marker(positions[0], [0.0, 1.0, 0.0])
    add_marker(positions[-1], [0.0, 0.0, 1.0])
    traj_pcd.points = o3d.utility.Vector3dVector(np.asarray(sampled, dtype=np.float64))
    traj_pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return traj_pcd


def write_pose_csv(path: Path, pose_rows: list[dict]) -> None:
    if pose_rows:
        pd.DataFrame(pose_rows).to_csv(path, index=False)


def visualize_with_camera(geometries, preset="top"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="XTEND reconstruction", width=1400, height=900)
    for geom in geometries:
        vis.add_geometry(geom)
    center = geometries[0].get_axis_aligned_bounding_box().get_center()
    ctr = vis.get_view_control()
    if preset == "top":
        ctr.set_front([0.0, 1.0, 0.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.45)
    elif preset == "front":
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.35)
    elif preset == "side":
        ctr.set_front([-1.0, 0.0, 0.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.45)
    else:
        raise ValueError(f"Unknown preset: {preset}")
    ctr.set_lookat(center)
    vis.run()
    vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse DA3 RGB-D frames and estimate smoothed velocity/distance.")
    parser.add_argument("--recording-dir", required=True)
    parser.add_argument("--camera-yaml", required=True)
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--rgb-dir", default="rgb_rectified")
    parser.add_argument("--depth-dir", default="depth_npy")
    parser.add_argument("--out-ply", default="gt_accumulated_cloud_yaml_yaw.ply")
    parser.add_argument("--out-ply-with-trajectory", default="gt_accumulated_cloud_with_trajectory.ply")
    parser.add_argument("--out-trajectory-ply", default="trajectory_lines.ply")
    parser.add_argument("--out-poses-csv", default="estimated_camera_poses.csv")
    parser.add_argument("--save-trajectory", action="store_true", default=True)
    parser.add_argument("--no-save-trajectory", dest="save_trajectory", action="store_false")
    parser.add_argument("--trajectory-sample-step-m", type=float, default=0.02)
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
    parser.add_argument("--yaw-axis", choices=["camera_y_up", "camera_y_down", "camera_z_forward", "camera_x_right"], default="camera_y_up")
    parser.add_argument("--yaw-sign", type=float, default=1.0)
    parser.add_argument("--init-mode", choices=["identity", "last", "yaw", "yaw_then_last"], default="yaw_then_last")
    parser.add_argument("--min-score", type=float, default=5e4)
    parser.add_argument("--max-rot-deg", type=float, default=10.0)
    parser.add_argument("--max-t-norm", type=float, default=0.8)
    parser.add_argument("--fallback-fps", type=float, default=20.0)
    parser.add_argument("--velocity-tau-sec", type=float, default=0.2)
    parser.add_argument("--use-smoothed-velocity-for-pose", action="store_true")
    parser.add_argument(
        "--measure-center-depth",
        action="store_true",
        help="Measure average depth in a square ROI at image center, slightly below middle.",
    )
    parser.add_argument(
        "--center-depth-roi-size-px",
        type=int,
        default=4,
        help="Square ROI size in pixels for center-depth measurement.",
    )
    parser.add_argument(
        "--center-depth-y-frac",
        type=float,
        default=0.44,
        help="Vertical ROI center as image-height fraction. 0.5=center, 0.58=below center.",
    )
    parser.add_argument(
        "--center-depth-min-m",
        type=float,
        default=0.2,
        help="Minimum valid depth for center-depth ROI.",
    )
    parser.add_argument(
        "--center-depth-max-m",
        type=float,
        default=15.0,
        help="Maximum valid depth for center-depth ROI.",
    )
    parser.add_argument(
        "--center-depth-debug-every",
        type=int,
        default=10,
        help="Save a debug RGB frame with the ROI rectangle every N target frames.",
    )
    parser.add_argument(
        "--center-depth-debug-dir",
        default="center_depth_debug",
        help="Directory for center-depth debug frames, relative to recording-dir unless absolute.",
    )

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization-preset", choices=["top", "front", "side"], default="top")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recording_dir = Path(args.recording_dir).expanduser()
    camera_yaml = Path(args.camera_yaml).expanduser()

    prefer_projection = args.prefer_projection_matrix and not args.use_camera_matrix
    yaml_width, yaml_height, K_raw, k_source = load_intrinsics_from_yaml(camera_yaml, prefer_projection)
    rgb_paths, depth_paths = find_frame_files(recording_dir, args.rgb_dir, args.depth_dir)

    first_bgr = cv2.imread(rgb_paths[0], cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise RuntimeError(f"Could not read first RGB image: {rgb_paths[0]}")
    image_h, image_w = first_bgr.shape[:2]
    K = scale_intrinsics_if_needed(K_raw, yaml_width, yaml_height, image_w, image_h)

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
        candidates = [recording_dir / "metadata.csv", recording_dir / "metadata_depth.csv"]
        metadata_path = next((p for p in candidates if p.exists()), None)

    if metadata_path and Path(metadata_path).exists():
        metadata_df = pd.read_csv(metadata_path)
        print(f"[info] loaded metadata: {metadata_path}")
        print(f"[info] metadata columns: {list(metadata_df.columns)}")
    else:
        print("[info] no metadata CSV loaded; using fallback FPS")

    option = o3d.pipelines.odometry.OdometryOption()
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([200, 100, 50, 20])
    option.depth_diff_max = 0.05
    option.depth_min = float(args.depth_min_m)
    option.depth_max = float(args.depth_max_m)

    accumulated_pcd = o3d.geometry.PointCloud()
    cam_to_world = np.eye(4, dtype=np.float64)
    last_transformation = np.eye(4, dtype=np.float64)

    pose_rows: list[dict] = []
    trajectory_positions: list[np.ndarray] = []
    total_distance_raw_m = 0.0
    total_distance_velocity_m = 0.0
    v_world_smooth: Optional[np.ndarray] = None

    start_idx = int(args.start_idx)
    end_idx = len(rgb_paths) - 1 if args.end_idx < 0 else min(int(args.end_idx), len(rgb_paths) - 1)
    step = max(int(args.step), 1)

    print(f"[info] fusing frame pairs: start={start_idx}, end={end_idx}, step={step}")
    print(f"[info] fallback_fps={args.fallback_fps}, velocity_tau_sec={args.velocity_tau_sec}")
    print(f"[info] use_smoothed_velocity_for_pose={args.use_smoothed_velocity_for_pose}")

    first_rgbd = make_rgbd_image(rgb_paths[start_idx], depth_paths[start_idx], args.depth_trunc_m)
    first_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(first_rgbd, pinhole)
    accumulated_pcd += first_pcd

    if args.measure_center_depth:
        first_depth_m = load_depth_m(depth_paths[start_idx])
        first_roi_stats = compute_center_depth_roi(
            depth_m=first_depth_m,
            rgb_shape=(image_h, image_w),
            roi_size_px=args.center_depth_roi_size_px,
            roi_center_y_frac=args.center_depth_y_frac,
            min_depth_m=args.center_depth_min_m,
            max_depth_m=args.center_depth_max_m,
        )
        center_debug_dir = Path(args.center_depth_debug_dir)
        if not center_debug_dir.is_absolute():
            center_debug_dir = recording_dir / center_debug_dir
        save_center_depth_debug_image(
            bgr_path=rgb_paths[start_idx],
            out_path=center_debug_dir / f"frame_{start_idx:06d}_center_depth.jpg",
            roi_stats=first_roi_stats,
            label=f"frame={start_idx}",
        )
    else:
        first_roi_stats = {
            "roi_x1": np.nan,
            "roi_y1": np.nan,
            "roi_x2": np.nan,
            "roi_y2": np.nan,
            "roi_valid_count": 0,
            "roi_total_count": 0,
            "roi_mean_depth_m": np.nan,
            "roi_median_depth_m": np.nan,
            "roi_std_depth_m": np.nan,
            "roi_min_depth_m": np.nan,
            "roi_max_depth_m": np.nan,
        }

    initial_pose = transform_to_pose_fields(cam_to_world)
    pose_rows.append({
        "frame_idx": start_idx,
        "rgb_path": rgb_paths[start_idx],
        "depth_path": depth_paths[start_idx],
        "accepted": True,
        "pair_src_idx": start_idx,
        "pair_tgt_idx": start_idx,
        "dt_sec": 0.0,
        "odom_score": np.nan,
        "odom_rot_deg": 0.0,
        "odom_t_norm_m": 0.0,
        "odom_tx_m": 0.0,
        "odom_ty_m": 0.0,
        "odom_tz_m": 0.0,
        "step_distance_raw_m": 0.0,
        "step_distance_velocity_m": 0.0,
        "total_distance_raw_m": 0.0,
        "total_distance_velocity_m": 0.0,
        "speed_inst_m_s": 0.0,
        "speed_smooth_m_s": 0.0,
        "v_world_inst_x_m_s": 0.0,
        "v_world_inst_y_m_s": 0.0,
        "v_world_inst_z_m_s": 0.0,
        "v_world_smooth_x_m_s": 0.0,
        "v_world_smooth_y_m_s": 0.0,
        "v_world_smooth_z_m_s": 0.0,
        "yaw_init_rot_deg": 0.0,
        "yaw_value": metadata_value(metadata_df, start_idx, args.yaw_column),
        **first_roi_stats,
        **initial_pose,
    })
    trajectory_positions.append(cam_to_world[:3, 3].copy())

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
            f"score={metrics['score']:.1f} rot={metrics['rot_deg']:.3f}deg t={metrics['t_norm']:.3f}m"
        )

        yaw_rot_deg = np.nan
        if yaw_init is not None:
            yaw_rot_deg = np.linalg.norm(Rot.from_matrix(yaw_init[:3, :3]).as_rotvec()) * 180.0 / np.pi

        dt = compute_dt_sec(metadata_df, src_idx, tgt_idx, args.fallback_fps)

        if accepted:
            last_transformation = trans
            prev_cam_to_world = cam_to_world.copy()
            candidate_cam_to_world = cam_to_world @ np.linalg.inv(trans)

            delta_world = candidate_cam_to_world[:3, 3] - prev_cam_to_world[:3, 3]
            v_world_inst = delta_world / dt
            speed_inst = float(np.linalg.norm(v_world_inst))

            v_world_smooth = update_velocity_ema(v_world_smooth, v_world_inst, dt, args.velocity_tau_sec)
            speed_smooth = float(np.linalg.norm(v_world_smooth))

            step_distance_raw = float(np.linalg.norm(delta_world))
            step_distance_velocity = speed_smooth * dt
            total_distance_raw_m += step_distance_raw
            total_distance_velocity_m += step_distance_velocity

            if args.use_smoothed_velocity_for_pose:
                cam_to_world = candidate_cam_to_world.copy()
                cam_to_world[:3, 3] = prev_cam_to_world[:3, 3] + v_world_smooth * dt
            else:
                cam_to_world = candidate_cam_to_world
        else:
            last_transformation = np.eye(4, dtype=np.float64)
            print(f"[pair {src_idx}->{tgt_idx}] rejected; global pose unchanged")
            v_world_inst = np.zeros(3, dtype=np.float64)
            if v_world_smooth is None:
                v_world_smooth = np.zeros(3, dtype=np.float64)
            speed_inst = 0.0
            speed_smooth = float(np.linalg.norm(v_world_smooth))
            step_distance_raw = 0.0
            step_distance_velocity = 0.0

        if args.measure_center_depth:
            depth_for_roi = load_depth_m(depth_paths[tgt_idx])
            roi_stats = compute_center_depth_roi(
                depth_m=depth_for_roi,
                rgb_shape=(image_h, image_w),
                roi_size_px=args.center_depth_roi_size_px,
                roi_center_y_frac=args.center_depth_y_frac,
                min_depth_m=args.center_depth_min_m,
                max_depth_m=args.center_depth_max_m,
            )

            if args.center_depth_debug_every > 0 and (tgt_idx - start_idx) % args.center_depth_debug_every == 0:
                center_debug_dir = Path(args.center_depth_debug_dir)
                if not center_debug_dir.is_absolute():
                    center_debug_dir = recording_dir / center_debug_dir
                save_center_depth_debug_image(
                    bgr_path=rgb_paths[tgt_idx],
                    out_path=center_debug_dir / f"frame_{tgt_idx:06d}_center_depth.jpg",
                    roi_stats=roi_stats,
                    label=f"frame={tgt_idx}",
                )

            print(
                f"[center-depth {tgt_idx}] "
                f"mean={roi_stats['roi_mean_depth_m']:.3f}m "
                f"median={roi_stats['roi_median_depth_m']:.3f}m "
                f"valid={roi_stats['roi_valid_count']}/{roi_stats['roi_total_count']} "
                f"roi=({roi_stats['roi_x1']},{roi_stats['roi_y1']})-"
                f"({roi_stats['roi_x2']},{roi_stats['roi_y2']})"
            )
        else:
            roi_stats = {
                "roi_x1": np.nan,
                "roi_y1": np.nan,
                "roi_x2": np.nan,
                "roi_y2": np.nan,
                "roi_valid_count": 0,
                "roi_total_count": 0,
                "roi_mean_depth_m": np.nan,
                "roi_median_depth_m": np.nan,
                "roi_std_depth_m": np.nan,
                "roi_min_depth_m": np.nan,
                "roi_max_depth_m": np.nan,
            }

        pose_fields = transform_to_pose_fields(cam_to_world)
        pose_rows.append({
            "frame_idx": tgt_idx,
            "rgb_path": rgb_paths[tgt_idx],
            "depth_path": depth_paths[tgt_idx],
            "accepted": bool(accepted),
            "pair_src_idx": src_idx,
            "pair_tgt_idx": tgt_idx,
            "dt_sec": float(dt),
            "odom_score": metrics["score"],
            "odom_rot_deg": metrics["rot_deg"],
            "odom_t_norm_m": metrics["t_norm"],
            "odom_tx_m": float(metrics["translation"][0]),
            "odom_ty_m": float(metrics["translation"][1]),
            "odom_tz_m": float(metrics["translation"][2]),
            "step_distance_raw_m": float(step_distance_raw),
            "step_distance_velocity_m": float(step_distance_velocity),
            "total_distance_raw_m": float(total_distance_raw_m),
            "total_distance_velocity_m": float(total_distance_velocity_m),
            "speed_inst_m_s": float(speed_inst),
            "speed_smooth_m_s": float(speed_smooth),
            "v_world_inst_x_m_s": float(v_world_inst[0]),
            "v_world_inst_y_m_s": float(v_world_inst[1]),
            "v_world_inst_z_m_s": float(v_world_inst[2]),
            "v_world_smooth_x_m_s": float(v_world_smooth[0]),
            "v_world_smooth_y_m_s": float(v_world_smooth[1]),
            "v_world_smooth_z_m_s": float(v_world_smooth[2]),
            **roi_stats,
            "yaw_init_rot_deg": yaw_rot_deg,
            "yaw_value": metadata_value(metadata_df, tgt_idx, args.yaw_column),
            **pose_fields,
        })
        trajectory_positions.append(cam_to_world[:3, 3].copy())

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
    print(f"[done] saved cloud: {out_ply}")
    print(f"[done] final points: {len(accumulated_pcd.points)}")

    out_poses_csv = Path(args.out_poses_csv)
    if not out_poses_csv.is_absolute():
        out_poses_csv = recording_dir / out_poses_csv
    write_pose_csv(out_poses_csv, pose_rows)
    print(f"[done] saved poses: {out_poses_csv}")

    if pose_rows:
        last = pose_rows[-1]
        print("[summary]")
        print(f"  total_distance_raw_m:      {last['total_distance_raw_m']:.3f}")
        print(f"  total_distance_velocity_m: {last['total_distance_velocity_m']:.3f}")
        print(f"  final cam_x_right_m:       {last['cam_x_right_m']:.3f}")
        print(f"  final cam_y_down_m:        {last['cam_y_down_m']:.3f}")
        print(f"  final cam_z_forward_m:     {last['cam_z_forward_m']:.3f}")

    trajectory_line_set = make_trajectory_line_set(trajectory_positions)
    trajectory_pcd = make_trajectory_point_cloud(trajectory_positions, line_step_m=float(args.trajectory_sample_step_m))

    if args.save_trajectory:
        out_trajectory_ply = Path(args.out_trajectory_ply)
        if not out_trajectory_ply.is_absolute():
            out_trajectory_ply = recording_dir / out_trajectory_ply
        if len(trajectory_positions) >= 2:
            o3d.io.write_line_set(str(out_trajectory_ply), trajectory_line_set)
            print(f"[done] saved trajectory lines: {out_trajectory_ply}")

        out_ply_with_trajectory = Path(args.out_ply_with_trajectory)
        if not out_ply_with_trajectory.is_absolute():
            out_ply_with_trajectory = recording_dir / out_ply_with_trajectory
        pcd_with_trajectory = accumulated_pcd + trajectory_pcd
        o3d.io.write_point_cloud(str(out_ply_with_trajectory), pcd_with_trajectory)
        print(f"[done] saved cloud with trajectory samples: {out_ply_with_trajectory}")

    if args.visualize:
        geoms = [accumulated_pcd]
        if len(trajectory_positions) >= 2:
            geoms.append(trajectory_line_set)
        visualize_with_camera(geoms, preset=args.visualization_preset)


if __name__ == "__main__":
    main()
