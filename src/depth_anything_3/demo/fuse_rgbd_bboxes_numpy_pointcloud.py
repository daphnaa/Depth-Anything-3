#!/usr/bin/env python3
"""
Pure NumPy/OpenCV point-cloud builder for XTEND + DA3 recordings.

Inputs:
  - per-frame JSON files with image name, pose/yaw, NanoOWL detections
  - RGB frames, usually resized to DA3/depth size
  - DA3 metric depth .npy files
  - camera YAML

Outputs:
  - scene_cloud.ply          full RGB-D cloud, optional
  - objects_cloud.ply        bbox/object-only cloud, optional
  - object_centers.ply       one marker sphere per detection
  - object_clusters.ply      clustered semantic landmarks
  - objects_3d.csv
  - object_clusters.csv
  - annotated_frames/*.jpg

This version does not use Open3D. It writes PLY files directly.
It also uses the correct optical-camera to body/world convention by default:
  camera optical: x right, y down, z forward
  body/world:     X forward, Y left, Z up
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


@dataclass
class Detection3D:
    label: str
    score: float
    image: str
    frame_idx: int
    center: np.ndarray
    count: int


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def matrix_from_yaml(data: dict[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(data[key]["data"], dtype=np.float64).reshape(shape)


def load_intrinsics_from_yaml(yaml_path: Path, prefer_projection: bool = True) -> tuple[int, int, np.ndarray, str]:
    data = load_yaml(yaml_path)
    width = int(data.get("image_width", data.get("width", 0)))
    height = int(data.get("image_height", data.get("height", 0)))

    if prefer_projection and "projection_matrix" in data:
        P = matrix_from_yaml(data, "projection_matrix", (3, 4))
        return width, height, P[:3, :3].copy(), "projection_matrix"
    if "camera_matrix" in data:
        return width, height, matrix_from_yaml(data, "camera_matrix", (3, 3)), "camera_matrix"
    if all(k in data for k in ["fx", "fy", "cx", "cy"]):
        K = np.array([
            [float(data["fx"]), 0.0, float(data["cx"])],
            [0.0, float(data["fy"]), float(data["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return width, height, K, "fx/fy/cx/cy"
    raise ValueError(f"Could not find camera intrinsics in {yaml_path}")


def scale_intrinsics_if_needed(K: np.ndarray, yaml_w: int, yaml_h: int, target_w: int, target_h: int) -> np.ndarray:
    out = K.copy().astype(np.float64)
    if yaml_w > 0 and yaml_h > 0 and (yaml_w, yaml_h) != (target_w, target_h):
        sx = target_w / float(yaml_w)
        sy = target_h / float(yaml_h)
        out[0, 0] *= sx
        out[1, 1] *= sy
        out[0, 2] *= sx
        out[1, 2] *= sy
    return out


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trajectory_by_image(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for item in data:
        image = item.get("image")
        pose = item.get("pose", {})
        if image and isinstance(pose, dict):
            out[Path(image).name] = {
                "x": float(pose.get("x", 0.0)),
                "y": float(pose.get("y", 0.0)),
                "z": float(pose.get("z", 0.0)),
                "yaw": float(pose.get("yaw", 0.0)),
            }
    return out


def get_pose(frame_json: dict[str, Any], image_name: str, trajectory_by_image: dict[str, dict[str, float]]) -> dict[str, float]:
    if image_name in trajectory_by_image:
        return trajectory_by_image[image_name]
    pose = frame_json.get("pose", {})
    return {
        "x": float(pose.get("x", 0.0)),
        "y": float(pose.get("y", 0.0)),
        "z": float(pose.get("z", 0.0)),
        "yaw": float(pose.get("yaw", 0.0)),
    }


def get_detections(frame_json: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    result = frame_json.get("nanoowl", {}).get("result", {})
    detections = result.get("detections", [])
    image_info = result.get("image", {})
    det_size = None
    if isinstance(image_info, dict) and "width" in image_info and "height" in image_info:
        det_size = (int(image_info["width"]), int(image_info["height"]))
    return detections, det_size


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def scale_bbox(bbox: list[float], from_w: int, from_h: int, to_w: int, to_h: int) -> tuple[int, int, int, int]:
    sx = to_w / float(from_w)
    sy = to_h / float(from_h)
    return clamp_bbox([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], to_w, to_h)


def shrink_bbox(bbox: tuple[int, int, int, int], width: int, height: int, shrink_frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    shrink = max(0.0, min(float(shrink_frac), 0.45))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(round(bw * shrink))
    dy = int(round(bh * shrink))
    return clamp_bbox([x1 + dx, y1 + dy, x2 - dx, y2 - dy], width, height)


def label_color_bgr(label: str) -> tuple[int, int, int]:
    seed = abs(hash(label)) % (2**32)
    rng = np.random.default_rng(seed)
    c = rng.integers(50, 235, size=3, dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def label_color_rgb(label: str) -> np.ndarray:
    b, g, r = label_color_bgr(label)
    return np.array([r, g, b], dtype=np.uint8)


def yaw_to_Rz(yaw_deg: float) -> np.ndarray:
    a = math.radians(float(yaw_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_Rxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_world_transform(
    pose: dict[str, float],
    yaw_sign: float,
    yaw_offset_deg: float,
    ignore_translation: bool,
    fixed_height_m: float,
    use_optical_to_body: bool,
    camera_roll_deg: float,
    camera_pitch_deg: float,
    camera_yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = yaw_sign * float(pose.get("yaw", 0.0)) + yaw_offset_deg
    R_world_from_body = yaw_to_Rz(yaw)

    if use_optical_to_body:
        R_body_from_cam = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)
    else:
        R_body_from_cam = np.eye(3, dtype=np.float64)

    R_mount = euler_Rxyz(camera_roll_deg, camera_pitch_deg, camera_yaw_deg)
    R = R_world_from_body @ R_mount @ R_body_from_cam

    if ignore_translation:
        t = np.array([0.0, 0.0, fixed_height_m], dtype=np.float64)
    else:
        t = np.array([
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", fixed_height_m)),
        ], dtype=np.float64)
    return R, t


def backproject_pixels(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape[:2]
    stride = max(1, int(pixel_stride))
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[0:h:stride, 0:w:stride]
    m = mask[0:h:stride, 0:w:stride] & np.isfinite(z) & (z > 0.0)
    if not np.any(m):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    u = xs[m].astype(np.float64)
    v = ys[m].astype(np.float64)
    z = z[m].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    colors_bgr = rgb[0:h:stride, 0:w:stride][m]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)
    return pts, colors_rgb


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    return (R @ points.T).T + t.reshape(1, 3)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None, ascii_format: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors mismatch: {points.shape[0]} vs {colors.shape[0]}")

    if ascii_format:
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        return

    vertex = np.empty(points.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


def make_marker_cloud(center: np.ndarray, color: np.ndarray, radius: float = 0.045, samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    pts = []
    for _ in range(samples):
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        if n > 1e-12:
            pts.append(center + radius * v / n)
    pts = np.asarray(pts, dtype=np.float64)
    colors = np.tile(color.reshape(1, 3), (pts.shape[0], 1)).astype(np.uint8)
    return pts, colors


def voxel_downsample_numpy(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    pts_sum = np.zeros((n, 3), dtype=np.float64)
    col_sum = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(pts_sum, inv, points)
    np.add.at(col_sum, inv, colors.astype(np.float64))
    np.add.at(counts, inv, 1.0)
    pts = pts_sum / counts[:, None]
    cols = np.clip(col_sum / counts[:, None], 0, 255).astype(np.uint8)
    return pts, cols


def greedy_cluster_by_label(detections: list[Detection3D], radius_m: float, min_count: int) -> list[dict[str, Any]]:
    clusters = []
    by_label: dict[str, list[Detection3D]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    cid = 0
    for label, items in by_label.items():
        unused = set(range(len(items)))
        while unused:
            seed_idx = unused.pop()
            seed = items[seed_idx]
            members = [seed_idx]
            changed = True
            center = seed.center.astype(np.float64).copy()
            while changed:
                changed = False
                for j in list(unused):
                    if np.linalg.norm(items[j].center - center) <= radius_m:
                        unused.remove(j)
                        members.append(j)
                        center = np.mean([items[k].center for k in members], axis=0)
                        changed = True
            if len(members) >= min_count:
                member_items = [items[k] for k in members]
                centers = np.stack([m.center for m in member_items], axis=0)
                scores = np.array([m.score for m in member_items], dtype=np.float64)
                clusters.append({
                    "cluster_id": cid,
                    "label": label,
                    "count": len(member_items),
                    "mean_score": float(scores.mean()),
                    "center": centers.mean(axis=0),
                })
                cid += 1
    return clusters


def main() -> None:
    p = argparse.ArgumentParser(description="Create pure NumPy point clouds from RGB/depth/bboxes.")
    p.add_argument("--json-dir", required=True)
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--depth-npy-dir", required=True)
    p.add_argument("--camera-yaml", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--trajectory-json", default="")
    p.add_argument("--json-glob", default="*.json")
    p.add_argument("--mode", choices=["scene", "objects", "both"], default="both")
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--min-depth-m", type=float, default=0.25)
    p.add_argument("--max-depth-m", type=float, default=3.0)
    p.add_argument("--pixel-stride", type=int, default=2)
    p.add_argument("--scene-pixel-stride", type=int, default=4)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--bbox-shrink", type=float, default=0.20)
    p.add_argument("--object-depth-band-m", type=float, default=0.15)
    p.add_argument("--min-object-points", type=int, default=30)
    p.add_argument("--yaw-sign", type=float, default=1.0)
    p.add_argument("--yaw-offset-deg", type=float, default=0.0)
    p.add_argument("--ignore-translation", action="store_true")
    p.add_argument("--fixed-height-m", type=float, default=0.83)
    p.add_argument("--no-optical-to-body", action="store_true")
    p.add_argument("--camera-roll-deg", type=float, default=0.0)
    p.add_argument("--camera-pitch-deg", type=float, default=0.0)
    p.add_argument("--camera-yaw-deg", type=float, default=0.0)
    p.add_argument("--use-label-color", action="store_true")
    p.add_argument("--cluster-objects", action="store_true")
    p.add_argument("--cluster-radius-m", type=float, default=0.45)
    p.add_argument("--cluster-min-count", type=int, default=2)
    p.add_argument("--ascii-ply", action="store_true")
    args = p.parse_args()

    json_dir = Path(args.json_dir).expanduser()
    rgb_dir = Path(args.rgb_dir).expanduser()
    depth_dir = Path(args.depth_npy_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ann_dir = out_dir / "annotated_frames"
    ann_dir.mkdir(parents=True, exist_ok=True)

    traj_path = Path(args.trajectory_json).expanduser() if args.trajectory_json else None
    trajectory_by_image = load_trajectory_by_image(traj_path)

    json_paths = sorted(json_dir.glob(args.json_glob))
    if not json_paths:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    # Determine fusion size from first available RGB/depth pair.
    first = read_json(json_paths[0])
    first_image = Path(first.get("image", json_paths[0].with_suffix(".jpg").name)).name
    first_rgb = cv2.imread(str(rgb_dir / first_image), cv2.IMREAD_COLOR)
    first_depth = np.load(depth_dir / f"{Path(first_image).stem}.npy")
    if first_rgb is None:
        raise RuntimeError(f"Could not read first RGB: {rgb_dir / first_image}")
    if first_depth.shape[:2] != first_rgb.shape[:2]:
        print(f"[warn] first depth size {first_depth.shape[:2]} != RGB size {first_rgb.shape[:2]}. Depth will be resized per frame.")
    target_h, target_w = first_depth.shape[:2]

    yaml_w, yaml_h, K_raw, k_source = load_intrinsics_from_yaml(Path(args.camera_yaml).expanduser(), prefer_projection=True)
    K = scale_intrinsics_if_needed(K_raw, yaml_w, yaml_h, target_w, target_h)
    print(f"[info] json frames: {len(json_paths)}")
    print(f"[info] target size: {target_w}x{target_h}")
    print(f"[info] intrinsics source: {k_source}, yaml size: {yaml_w}x{yaml_h}")
    print(f"[info] K:\n{K}")
    print(f"[info] optical_to_body={not args.no_optical_to_body}")

    scene_pts_all, scene_cols_all = [], []
    obj_pts_all, obj_cols_all = [], []
    center_pts_all, center_cols_all = [], []
    detections3d: list[Detection3D] = []

    csv_path = out_dir / "objects_3d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "frame_idx", "image", "label", "score", "point_count",
            "center_x_m", "center_y_m", "center_z_m",
            "bbox_det_x1", "bbox_det_y1", "bbox_det_x2", "bbox_det_y2",
            "bbox_depth_x1", "bbox_depth_y1", "bbox_depth_x2", "bbox_depth_y2",
            "yaw_deg",
        ])

        for frame_idx, json_path in enumerate(json_paths):
            if frame_idx % max(1, args.frame_step) != 0:
                continue
            frame_json = read_json(json_path)
            image_name = Path(frame_json.get("image", json_path.with_suffix(".jpg").name)).name
            stem = Path(image_name).stem
            rgb_path = rgb_dir / image_name
            depth_path = depth_dir / f"{stem}.npy"
            if not rgb_path.exists() or not depth_path.exists():
                print(f"[warn] missing RGB/depth for {image_name}")
                continue
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_m = np.load(depth_path).astype(np.float32)
            if rgb is None:
                print(f"[warn] failed to read {rgb_path}")
                continue
            if depth_m.ndim == 3:
                depth_m = depth_m[..., 0]
            if depth_m.shape[:2] != (target_h, target_w):
                depth_m = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if rgb.shape[:2] != (target_h, target_w):
                rgb_for_cloud = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                rgb_for_cloud = rgb

            valid_depth = np.isfinite(depth_m) & (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m)
            pose = get_pose(frame_json, image_name, trajectory_by_image)
            R, t = make_world_transform(
                pose=pose,
                yaw_sign=args.yaw_sign,
                yaw_offset_deg=args.yaw_offset_deg,
                ignore_translation=args.ignore_translation,
                fixed_height_m=args.fixed_height_m,
                use_optical_to_body=not args.no_optical_to_body,
                camera_roll_deg=args.camera_roll_deg,
                camera_pitch_deg=args.camera_pitch_deg,
                camera_yaw_deg=args.camera_yaw_deg,
            )

            if args.mode in ("scene", "both"):
                pts_cam, cols = backproject_pixels(depth_m, rgb_for_cloud, K, valid_depth, args.scene_pixel_stride)
                pts_world = transform_points(pts_cam, R, t)
                scene_pts_all.append(pts_world)
                scene_cols_all.append(cols)

            detections, det_size = get_detections(frame_json)
            if det_size is None:
                det_size = (rgb.shape[1], rgb.shape[0])
            det_w, det_h = det_size
            annotated = rgb_for_cloud.copy()

            for det in detections:
                score = float(det.get("score", 0.0))
                label = str(det.get("label", "object"))
                bbox = det.get("bbox")
                if bbox is None or score < args.min_score:
                    continue

                bbox_depth = scale_bbox(bbox, det_w, det_h, target_w, target_h)
                bbox_depth = shrink_bbox(bbox_depth, target_w, target_h, args.bbox_shrink)
                x1, y1, x2, y2 = bbox_depth

                patch = depth_m[y1:y2, x1:x2]
                patch_valid = np.isfinite(patch) & (patch >= args.min_depth_m) & (patch <= args.max_depth_m)
                vals = patch[patch_valid]
                if vals.size < args.min_object_points:
                    continue
                median_z = float(np.median(vals))
                obj_mask = np.zeros((target_h, target_w), dtype=bool)
                local_mask = np.zeros_like(patch, dtype=bool)
                local_mask[patch_valid] = np.abs(patch[patch_valid] - median_z) <= args.object_depth_band_m
                obj_mask[y1:y2, x1:x2] = local_mask
                obj_mask &= valid_depth

                pts_cam, cols = backproject_pixels(depth_m, rgb_for_cloud, K, obj_mask, args.pixel_stride)
                if pts_cam.shape[0] < args.min_object_points:
                    continue
                pts_world = transform_points(pts_cam, R, t)
                center = np.median(pts_world, axis=0)

                if args.use_label_color:
                    color = label_color_rgb(label)
                    cols = np.tile(color.reshape(1, 3), (pts_world.shape[0], 1))
                else:
                    color = label_color_rgb(label)

                if args.mode in ("objects", "both"):
                    obj_pts_all.append(pts_world)
                    obj_cols_all.append(cols)

                marker_pts, marker_cols = make_marker_cloud(center, color)
                center_pts_all.append(marker_pts)
                center_cols_all.append(marker_cols)

                detections3d.append(Detection3D(label=label, score=score, image=image_name, frame_idx=frame_idx, center=center, count=pts_world.shape[0]))
                writer.writerow([
                    frame_idx, image_name, label, score, int(pts_world.shape[0]),
                    float(center[0]), float(center[1]), float(center[2]),
                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                    x1, y1, x2, y2,
                    float(pose.get("yaw", 0.0)),
                ])

                bgr = label_color_bgr(label)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(annotated, f"{label} {score:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)

            cv2.imwrite(str(ann_dir / image_name), annotated)
            if frame_idx % 20 == 0:
                print(f"[frame {frame_idx:05d}] {image_name} yaw={pose.get('yaw', 0.0):.2f} detections_so_far={len(detections3d)}")

    if scene_pts_all:
        pts = np.concatenate(scene_pts_all, axis=0)
        cols = np.concatenate(scene_cols_all, axis=0)
        pts, cols = voxel_downsample_numpy(pts, cols, args.voxel_size)
        write_ply(out_dir / "scene_cloud.ply", pts, cols, ascii_format=args.ascii_ply)
        print(f"[done] scene_cloud.ply points={len(pts)}")

    if obj_pts_all:
        pts = np.concatenate(obj_pts_all, axis=0)
        cols = np.concatenate(obj_cols_all, axis=0)
        pts, cols = voxel_downsample_numpy(pts, cols, args.voxel_size)
        write_ply(out_dir / "objects_cloud.ply", pts, cols, ascii_format=args.ascii_ply)
        print(f"[done] objects_cloud.ply points={len(pts)}")

    if center_pts_all:
        pts = np.concatenate(center_pts_all, axis=0)
        cols = np.concatenate(center_cols_all, axis=0)
        write_ply(out_dir / "object_centers.ply", pts, cols, ascii_format=args.ascii_ply)
        print(f"[done] object_centers.ply markers={len(detections3d)}")

    if args.cluster_objects and detections3d:
        clusters = greedy_cluster_by_label(detections3d, args.cluster_radius_m, args.cluster_min_count)
        with open(out_dir / "object_clusters.csv", "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["cluster_id", "label", "count", "mean_score", "center_x_m", "center_y_m", "center_z_m"])
            for c in clusters:
                center = c["center"]
                writer.writerow([c["cluster_id"], c["label"], c["count"], c["mean_score"], center[0], center[1], center[2]])

        cluster_pts, cluster_cols = [], []
        for c in clusters:
            color = label_color_rgb(c["label"])
            pts, cols = make_marker_cloud(c["center"], color, radius=0.075, samples=180)
            cluster_pts.append(pts)
            cluster_cols.append(cols)
        if cluster_pts:
            write_ply(out_dir / "object_clusters.ply", np.concatenate(cluster_pts), np.concatenate(cluster_cols), ascii_format=args.ascii_ply)
        print(f"[done] object clusters={len(clusters)}")

    print(f"[done] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
