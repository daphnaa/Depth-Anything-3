#!/usr/bin/env python3
"""
Fuse RGB + metric depth .npy + known x/y/z/yaw trajectory into a colored Open3D point cloud.

Intended for a 360 turn in a room:
- RGB images are on disk.
- DA3 metric depth is saved as .npy in meters.
- Trajectory JSON contains entries like:
    {"image": "R2_...jpg", "pose": {"x": ..., "y": ..., "z": ..., "yaw": ...}}

Default coordinate convention:
- Camera optical frame: X right, Y down, Z forward.
- Body/world navigation frame: X forward, Y left, Z up.
- yaw rotates body X/Y around world Z.

If the cloud rotates in the wrong direction, use --yaw-sign -1.
If the room is globally rotated, tune --yaw-offset-deg.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def matrix_from_yaml(data: dict[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(data[key]["data"], dtype=np.float64).reshape(shape)


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
    elif all(k in data for k in ("fx", "fy", "cx", "cy")):
        K = np.array(
            [[float(data["fx"]), 0.0, float(data["cx"])],
             [0.0, float(data["fy"]), float(data["cy"])],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        source = "fx/fy/cx/cy"
    else:
        raise ValueError(
            f"Could not load intrinsics from {yaml_path}. "
            "Expected projection_matrix, camera_matrix, or fx/fy/cx/cy."
        )
    return width, height, K, source


def scale_intrinsics(K: np.ndarray, from_w: int, from_h: int, to_w: int, to_h: int) -> np.ndarray:
    out = K.copy().astype(np.float64)
    if from_w > 0 and from_h > 0 and (from_w != to_w or from_h != to_h):
        sx = float(to_w) / float(from_w)
        sy = float(to_h) / float(from_h)
        out[0, 0] *= sx
        out[1, 1] *= sy
        out[0, 2] *= sx
        out[1, 2] *= sy
    return out


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    if not isinstance(data, list):
        raise ValueError("Trajectory JSON must be a list, or a dict with a 'frames' list.")

    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "pose" not in item:
            continue
        image = item.get("image") or item.get("rgb") or item.get("filename") or f"frame_{i:06d}.jpg"
        out.append({"image": str(image), "pose": item["pose"]})
    if not out:
        raise ValueError(f"No trajectory entries with pose found in {path}")
    return out


def find_image(rgb_dir: Path, image_name: str) -> Path:
    p = rgb_dir / image_name
    if p.exists():
        return p
    stem = Path(image_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find RGB image for trajectory entry: {image_name}")


def find_depth(depth_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    p = depth_dir / f"{stem}.npy"
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find depth .npy matching image stem: {stem}")


def clean_depth(depth_m: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        else:
            raise ValueError("Depth .npy has 3 channels. Expected HxW metric depth.")
    depth[~np.isfinite(depth)] = 0.0

    valid = depth[depth > 0.0]
    if valid.size > 0 and (float(np.median(valid)) > 100.0 or float(np.max(valid)) > 100.0):
        depth = depth / 1000.0

    depth[depth < float(min_depth_m)] = 0.0
    depth[depth > float(max_depth_m)] = 0.0
    return depth.astype(np.float32)


def backproject_depth(depth_m: np.ndarray, rgb: np.ndarray, K: np.ndarray, pixel_stride: int):
    h, w = depth_m.shape[:2]
    stride = max(1, int(pixel_stride))
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[ys, xs]
    valid = z > 0.0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    u = xs[valid].astype(np.float64)
    v = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_optical = np.stack([x, y, z], axis=1)

    rgb_h, rgb_w = rgb.shape[:2]
    if (rgb_w, rgb_h) != (w, h):
        color_u = np.clip(np.round(u * (rgb_w / float(w))).astype(np.int32), 0, rgb_w - 1)
        color_v = np.clip(np.round(v * (rgb_h / float(h))).astype(np.int32), 0, rgb_h - 1)
    else:
        color_u = np.clip(np.round(u).astype(np.int32), 0, rgb_w - 1)
        color_v = np.clip(np.round(v).astype(np.int32), 0, rgb_h - 1)
    colors = rgb[color_v, color_u].astype(np.float64) / 255.0
    return pts_optical, colors


def yaw_to_rotation_z(yaw_deg: float, yaw_sign: float, yaw_offset_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_offset_deg) + float(yaw_sign) * float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_world_from_pose(pose: dict[str, Any], yaw_sign: float, yaw_offset_deg: float,
                         ignore_translation: bool, fixed_height_m: float | None) -> np.ndarray:
    yaw = float(pose.get("yaw", 0.0))
    R_world_body = yaw_to_rotation_z(yaw, yaw_sign=yaw_sign, yaw_offset_deg=yaw_offset_deg)

    # optical [right, down, forward] -> body [forward, left, up]
    R_body_optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_world_body @ R_body_optical

    if ignore_translation:
        tx, ty, tz = 0.0, 0.0, 0.0
    else:
        tx = float(pose.get("x", 0.0))
        ty = float(pose.get("y", 0.0))
        tz = float(pose.get("z", 0.0))
    if fixed_height_m is not None:
        tz = float(fixed_height_m)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def make_trajectory_lineset(positions: list[np.ndarray]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    if not positions:
        return ls
    pts = np.asarray(positions, dtype=np.float64)
    ls.points = o3d.utility.Vector3dVector(pts)
    if len(positions) >= 2:
        lines = np.asarray([[i, i + 1] for i in range(len(positions) - 1)], dtype=np.int32)
        ls.lines = o3d.utility.Vector2iVector(lines)
    return ls


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse a 360 room turn from RGB + depth_npy + known x/y/z/yaw trajectory.")
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--depth-npy-dir", required=True)
    parser.add_argument("--camera-yaml", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--out-trajectory-ply", default="")
    parser.add_argument("--prefer-projection-matrix", action="store_true", default=True)
    parser.add_argument("--use-camera-matrix", action="store_true")
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--yaw-sign", type=float, default=1.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--ignore-translation", action="store_true")
    parser.add_argument("--fixed-height-m", type=float, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory_json).expanduser()
    rgb_dir = Path(args.rgb_dir).expanduser()
    depth_dir = Path(args.depth_npy_dir).expanduser()
    camera_yaml = Path(args.camera_yaml).expanduser()
    out_ply = Path(args.out_ply).expanduser()

    yaml_w, yaml_h, K_raw, source = load_intrinsics_from_yaml(camera_yaml, prefer_projection=not args.use_camera_matrix)
    trajectory = load_trajectory(trajectory_path)[::max(1, int(args.frame_step))]
    if args.max_frames > 0:
        trajectory = trajectory[:int(args.max_frames)]

    print(f"[info] frames to fuse: {len(trajectory)}")
    print(f"[info] intrinsics source: {source}, yaml size: {yaml_w}x{yaml_h}")
    print(f"[info] yaw_sign={args.yaw_sign}, yaw_offset_deg={args.yaw_offset_deg}")

    all_points = []
    all_colors = []
    trajectory_positions = []

    for idx, entry in enumerate(trajectory):
        image_name = entry["image"]
        pose = entry["pose"]
        rgb_path = find_image(rgb_dir, image_name)
        depth_path = find_depth(depth_dir, image_name)

        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read RGB: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_m = clean_depth(np.load(depth_path), args.min_depth_m, args.max_depth_m)

        depth_h, depth_w = depth_m.shape[:2]
        K = scale_intrinsics(K_raw, yaml_w, yaml_h, depth_w, depth_h)
        pts_optical, colors = backproject_depth(depth_m, rgb, K, args.pixel_stride)
        if pts_optical.shape[0] == 0:
            print(f"[warn] empty valid depth: {depth_path.name}")
            continue

        T_world_optical = make_world_from_pose(
            pose, args.yaw_sign, args.yaw_offset_deg, args.ignore_translation, args.fixed_height_m
        )
        pts_world = (T_world_optical[:3, :3] @ pts_optical.T).T + T_world_optical[:3, 3]
        all_points.append(pts_world)
        all_colors.append(colors)
        trajectory_positions.append(T_world_optical[:3, 3].copy())

        if idx % 20 == 0:
            valid = depth_m[depth_m > 0.0]
            med = float(np.median(valid)) if valid.size else float("nan")
            print(f"[frame {idx:05d}] {Path(image_name).name} points={pts_world.shape[0]} depth_med={med:.3f}m yaw={float(pose.get('yaw', 0.0)):.2f}")

    if not all_points:
        raise RuntimeError("No points were generated.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0).astype(np.float64))

    if args.voxel_size > 0.0:
        print(f"[info] before voxel: {len(pcd.points)} points")
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))
        print(f"[info] after voxel:  {len(pcd.points)} points")

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"[done] saved cloud: {out_ply}")

    geoms = [pcd]
    if args.out_trajectory_ply:
        traj = make_trajectory_lineset(trajectory_positions)
        traj_path = Path(args.out_trajectory_ply).expanduser()
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_line_set(str(traj_path), traj)
        print(f"[done] saved trajectory: {traj_path}")
        geoms.append(traj)

    if args.visualize:
        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
