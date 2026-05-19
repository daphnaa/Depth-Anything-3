#!/usr/bin/env python3
"""
Create annotated frames and an object-only 3D point cloud from:
- RGB images
- DA3 metric depth .npy files
- per-frame JSON files with NanoOWL detections
- camera YAML
- pose/yaw either in each JSON or in a trajectory JSON

This is meant for XTEND/DA3 recordings where detections are in the original
RGB resolution, while depth may be in a different model output resolution.
The script scales bbox coordinates to the depth/RGB fusion resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import yaml


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
        raise ValueError(f"Could not find intrinsics in YAML: {yaml_path}")

    return width, height, K, source


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


def safe_stem_from_image_name(image_name: str) -> str:
    return Path(image_name).stem


def collect_jsons(json_dir: Path, json_glob: str) -> list[Path]:
    paths = sorted(json_dir.glob(json_glob))
    if not paths:
        raise RuntimeError(f"No JSON files found in {json_dir} with glob {json_glob}")
    return paths


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
    if isinstance(image_info, dict) and "width" in image_info and "height" in image_info:
        det_size = (int(image_info["width"]), int(image_info["height"]))
    else:
        det_size = None
    return detections, det_size


def label_color(label: str) -> tuple[int, int, int]:
    # Deterministic BGR color for OpenCV annotation.
    seed = abs(hash(label)) % (2**32)
    rng = np.random.default_rng(seed)
    color = rng.integers(40, 230, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def label_color_rgb_float(label: str) -> np.ndarray:
    b, g, r = label_color(label)
    return np.array([r, g, b], dtype=np.float64) / 255.0


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
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


def scale_bbox(
    bbox: list[float],
    from_w: int,
    from_h: int,
    to_w: int,
    to_h: int,
) -> tuple[int, int, int, int]:
    sx = to_w / float(from_w)
    sy = to_h / float(from_h)
    scaled = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
    return clamp_bbox(scaled, to_w, to_h)


def shrink_bbox_pixels(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    shrink_frac: float,
) -> tuple[int, int, int, int]:
    """Shrink a bbox around its center by a fractional margin on each side."""
    x1, y1, x2, y2 = bbox
    shrink = max(0.0, min(float(shrink_frac), 0.45))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(round(bw * shrink))
    dy = int(round(bh * shrink))
    return clamp_bbox([x1 + dx, y1 + dy, x2 - dx, y2 - dy], width, height)


def robust_depth_for_bbox(
    depth_m: np.ndarray,
    bbox_depth: tuple[int, int, int, int],
    min_depth_m: float,
    max_depth_m: float,
) -> float | None:
    """Return a robust depth estimate from the inner bbox region."""
    x1, y1, x2, y2 = bbox_depth
    patch = depth_m[y1:y2, x1:x2].astype(np.float32, copy=False)
    valid = np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
    vals = patch[valid]
    if vals.size < 5:
        return None
    return float(np.median(vals))


def make_sphere_cloud(center: np.ndarray, color: np.ndarray, radius: float = 0.05, samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Create a small colored sphere-like marker as points."""
    rng = np.random.default_rng(1234)
    pts = []
    for _ in range(samples):
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        if n < 1e-12:
            continue
        pts.append(center + radius * v / n)
    pts = np.asarray(pts, dtype=np.float64)
    colors = np.tile(color.reshape(1, 3), (pts.shape[0], 1))
    return pts, colors


def save_object_centers_ply(rows: list[dict[str, Any]], out_path: Path, marker_radius: float = 0.05) -> None:
    if not rows:
        return
    pts_all = []
    cols_all = []
    for row in rows:
        center = np.array([row["center_x_m"], row["center_y_m"], row["center_z_m"]], dtype=np.float64)
        color = label_color_rgb_float(str(row["label"]))
        pts, cols = make_sphere_cloud(center, color, radius=marker_radius)
        pts_all.append(pts)
        cols_all.append(cols)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    o3d.io.write_point_cloud(str(out_path), pcd)


def cluster_object_centers(rows: list[dict[str, Any]], cluster_radius_m: float, min_cluster_size: int) -> list[dict[str, Any]]:
    """Simple per-label clustering of object centers using Open3D DBSCAN."""
    if not rows:
        return []
    clusters: list[dict[str, Any]] = []
    labels = sorted(set(str(r["label"]) for r in rows))
    for label in labels:
        label_rows = [r for r in rows if str(r["label"]) == label]
        centers = np.array([[r["center_x_m"], r["center_y_m"], r["center_z_m"]] for r in label_rows], dtype=np.float64)
        if centers.shape[0] == 0:
            continue
        if centers.shape[0] < max(1, min_cluster_size):
            # Keep singletons only if requested min_cluster_size <= 1.
            if min_cluster_size <= 1:
                for i, c in enumerate(centers):
                    clusters.append({
                        "cluster_id": len(clusters),
                        "label": label,
                        "count": 1,
                        "mean_score": float(label_rows[i].get("score", 0.0)),
                        "center_x_m": float(c[0]),
                        "center_y_m": float(c[1]),
                        "center_z_m": float(c[2]),
                    })
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        db = np.asarray(pcd.cluster_dbscan(eps=float(cluster_radius_m), min_points=int(min_cluster_size), print_progress=False))
        for cid in sorted(set(int(x) for x in db if x >= 0)):
            idx = np.where(db == cid)[0]
            c = np.median(centers[idx], axis=0)
            scores = [float(label_rows[i].get("score", 0.0)) for i in idx]
            clusters.append({
                "cluster_id": len(clusters),
                "label": label,
                "count": int(len(idx)),
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "center_x_m": float(c[0]),
                "center_y_m": float(c[1]),
                "center_z_m": float(c[2]),
            })
    return clusters


def save_clusters_csv(clusters: list[dict[str, Any]], out_path: Path) -> None:
    if not clusters:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clusters[0].keys()))
        writer.writeheader()
        writer.writerows(clusters)


def save_clusters_ply(clusters: list[dict[str, Any]], out_path: Path, marker_radius: float = 0.08) -> None:
    if not clusters:
        return
    pts_all = []
    cols_all = []
    for row in clusters:
        center = np.array([row["center_x_m"], row["center_y_m"], row["center_z_m"]], dtype=np.float64)
        color = label_color_rgb_float(str(row["label"]))
        pts, cols = make_sphere_cloud(center, color, radius=marker_radius, samples=180)
        pts_all.append(pts)
        cols_all.append(cols)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    o3d.io.write_point_cloud(str(out_path), pcd)


def camera_points_to_world(
    points_cam: np.ndarray,
    pose: dict[str, float],
    yaw_sign: float,
    yaw_offset_deg: float,
    ignore_translation: bool,
    fixed_height_m: float | None,
) -> np.ndarray:
    # Camera optical frame: X right, Y down, Z forward.
    # World frame used here: X forward, Y left, Z up.
    x_right = points_cam[:, 0]
    y_down = points_cam[:, 1]
    z_forward = points_cam[:, 2]

    points_body = np.column_stack([z_forward, -x_right, -y_down])

    yaw_deg = yaw_sign * float(pose.get("yaw", 0.0)) + yaw_offset_deg
    yaw_rad = math.radians(yaw_deg)
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    Rz = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    points_world = points_body @ Rz.T

    if ignore_translation:
        t = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if fixed_height_m is not None:
            t[2] = float(fixed_height_m)
    else:
        t = np.array(
            [
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        if fixed_height_m is not None:
            t[2] = float(fixed_height_m)

    return points_world + t[None, :]


def backproject_bbox_points(
    depth_m: np.ndarray,
    rgb_for_depth: np.ndarray,
    bbox_depth: tuple[int, int, int, int],
    K: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    use_label_color: bool,
    label: str,
    object_depth_m: float | None = None,
    object_depth_band_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_depth
    ys = np.arange(y1, y2, max(1, stride), dtype=np.int32)
    xs = np.arange(x1, x2, max(1, stride), dtype=np.int32)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[grid_y, grid_x].astype(np.float64)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if object_depth_m is not None and object_depth_band_m > 0.0:
        valid &= np.abs(z - float(object_depth_m)) <= float(object_depth_band_m)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3))

    u = grid_x[valid].astype(np.float64)
    v = grid_y[valid].astype(np.float64)
    z = z[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.column_stack([x, y, z])

    if use_label_color:
        rgb = np.tile(label_color_rgb_float(label), (pts_cam.shape[0], 1))
    else:
        bgr_vals = rgb_for_depth[grid_y[valid], grid_x[valid]].astype(np.float64) / 255.0
        rgb = bgr_vals[:, ::-1]

    return pts_cam, rgb


def annotate_image(
    bgr: np.ndarray,
    detections: list[dict[str, Any]],
    min_score: float,
    out_path: Path,
    det_w: int,
    det_h: int,
) -> None:
    """Draw detections on the displayed RGB image.

    NanoOWL bboxes are often in the original image coordinates, for example
    720x420, while the RGB image used for fusion may be resized to 504x392.
    Therefore bbox coordinates must be scaled to the current image size before
    drawing.
    """
    canvas = bgr.copy()
    h, w = canvas.shape[:2]
    for det in detections:
        score = float(det.get("score", 0.0))
        if score < min_score:
            continue
        label = str(det.get("label", "object"))
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = scale_bbox(bbox, det_w, det_h, w, h)
        color = label_color(label)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        y_text = max(18, y1 - 6)
        cv2.putText(canvas, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw bbox annotations and create object-only point cloud from DA3 depth.")
    parser.add_argument("--json-dir", required=True, help="Folder with per-frame JSON files.")
    parser.add_argument("--rgb-dir", required=True, help="Folder with RGB images.")
    parser.add_argument("--depth-npy-dir", required=True, help="Folder with metric depth .npy files.")
    parser.add_argument("--camera-yaml", required=True, help="Camera YAML matching the depth/RGB fusion resolution.")
    parser.add_argument("--out-dir", required=True, help="Output folder.")
    parser.add_argument("--trajectory-json", default="", help="Optional trajectory JSON keyed by image name.")
    parser.add_argument("--json-glob", default="*.json")
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--yaw-sign", type=float, default=1.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--ignore-translation", action="store_true")
    parser.add_argument("--fixed-height-m", type=float, default=None)
    parser.add_argument("--use-label-color", action="store_true", help="Color object points by label instead of RGB texture.")
    parser.add_argument("--prefer-camera-matrix", action="store_true", help="Use camera_matrix instead of projection_matrix.")
    parser.add_argument("--bbox-shrink", type=float, default=0.0, help="Shrink each bbox by this fraction on every side before backprojection, e.g. 0.15.")
    parser.add_argument("--object-depth-band-m", type=float, default=0.0, help="Keep only bbox pixels within this distance from the robust bbox median depth. 0 disables.")
    parser.add_argument("--min-object-points", type=int, default=1, help="Drop detections that produce fewer than this many 3D points.")
    parser.add_argument("--save-object-centers-ply", action="store_true", help="Save object center markers as a PLY.")
    parser.add_argument("--cluster-objects", action="store_true", help="Cluster object centers per label and save clustered landmark outputs.")
    parser.add_argument("--cluster-radius-m", type=float, default=0.45, help="DBSCAN radius for object-center clustering.")
    parser.add_argument("--cluster-min-count", type=int, default=2, help="Minimum detections per object cluster.")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    json_dir = Path(args.json_dir).expanduser()
    rgb_dir = Path(args.rgb_dir).expanduser()
    depth_dir = Path(args.depth_npy_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ann_dir = out_dir / "annotated_frames"
    debug_dir = out_dir / "debug_object_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    traj_path = Path(args.trajectory_json).expanduser() if args.trajectory_json else None
    trajectory_by_image = load_trajectory_by_image(traj_path)

    yaml_w, yaml_h, K_raw, k_source = load_intrinsics_from_yaml(
        Path(args.camera_yaml).expanduser(),
        prefer_projection=not args.prefer_camera_matrix,
    )

    json_paths = collect_jsons(json_dir, args.json_glob)
    json_paths = json_paths[:: max(1, int(args.frame_step))]

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    object_rows: list[dict[str, Any]] = []

    print(f"[info] json frames: {len(json_paths)}")
    print(f"[info] intrinsics source: {k_source}, yaml size: {yaml_w}x{yaml_h}")

    for frame_idx, json_path in enumerate(json_paths):
        frame_json = read_json(json_path)
        image_name = Path(frame_json.get("image", json_path.with_suffix(".jpg").name)).name
        stem = safe_stem_from_image_name(image_name)

        rgb_path = rgb_dir / image_name
        if not rgb_path.exists():
            candidates = list(rgb_dir.glob(stem + ".*"))
            if candidates:
                rgb_path = candidates[0]
            else:
                print(f"[warn] missing RGB for {image_name}")
                continue

        depth_path = depth_dir / f"{stem}.npy"
        if not depth_path.exists():
            print(f"[warn] missing depth for {image_name}: {depth_path}")
            continue

        bgr_orig = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr_orig is None:
            print(f"[warn] failed to read RGB: {rgb_path}")
            continue
        orig_h, orig_w = bgr_orig.shape[:2]

        depth_m = np.load(depth_path).astype(np.float32)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        depth_h, depth_w = depth_m.shape[:2]

        K = scale_intrinsics_if_needed(K_raw, yaml_w, yaml_h, depth_w, depth_h)
        bgr_depth = cv2.resize(bgr_orig, (depth_w, depth_h), interpolation=cv2.INTER_AREA)

        detections, det_size = get_detections(frame_json)
        if det_size is None:
            det_w, det_h = orig_w, orig_h
        else:
            det_w, det_h = det_size

        annotate_image(bgr_orig, detections, args.min_score, ann_dir / f"{stem}_bbox.jpg", det_w, det_h)

        pose = get_pose(frame_json, image_name, trajectory_by_image)
        frame_points: list[np.ndarray] = []
        frame_colors: list[np.ndarray] = []

        for det_idx, det in enumerate(detections):
            score = float(det.get("score", 0.0))
            if score < args.min_score:
                continue
            label = str(det.get("label", "object"))
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            bbox_depth = scale_bbox(bbox, det_w, det_h, depth_w, depth_h)
            bbox_depth = shrink_bbox_pixels(bbox_depth, depth_w, depth_h, args.bbox_shrink)
            object_depth = None
            if args.object_depth_band_m > 0.0:
                object_depth = robust_depth_for_bbox(
                    depth_m=depth_m,
                    bbox_depth=bbox_depth,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                )

            pts_cam, colors = backproject_bbox_points(
                depth_m=depth_m,
                rgb_for_depth=bgr_depth,
                bbox_depth=bbox_depth,
                K=K,
                stride=args.pixel_stride,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                use_label_color=args.use_label_color,
                label=label,
                object_depth_m=object_depth,
                object_depth_band_m=args.object_depth_band_m,
            )
            if pts_cam.shape[0] < int(args.min_object_points):
                continue

            pts_world = camera_points_to_world(
                pts_cam,
                pose=pose,
                yaw_sign=args.yaw_sign,
                yaw_offset_deg=args.yaw_offset_deg,
                ignore_translation=args.ignore_translation,
                fixed_height_m=args.fixed_height_m,
            )

            all_points.append(pts_world)
            all_colors.append(colors)
            frame_points.append(pts_world)
            frame_colors.append(colors)

            center = np.median(pts_world, axis=0)
            object_rows.append(
                {
                    "frame_idx": frame_idx,
                    "image": image_name,
                    "label": label,
                    "score": score,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "points": int(pts_world.shape[0]),
                    "object_depth_m": float(object_depth) if object_depth is not None else float("nan"),
                    "bbox_depth_x1": int(bbox_depth[0]),
                    "bbox_depth_y1": int(bbox_depth[1]),
                    "bbox_depth_x2": int(bbox_depth[2]),
                    "bbox_depth_y2": int(bbox_depth[3]),
                    "center_x_m": float(center[0]),
                    "center_y_m": float(center[1]),
                    "center_z_m": float(center[2]),
                    "pose_x": float(pose.get("x", 0.0)),
                    "pose_y": float(pose.get("y", 0.0)),
                    "pose_z": float(pose.get("z", 0.0)),
                    "pose_yaw_deg": float(pose.get("yaw", 0.0)),
                }
            )

        if frame_points:
            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(np.vstack(frame_points))
            pcd_frame.colors = o3d.utility.Vector3dVector(np.vstack(frame_colors))
            o3d.io.write_point_cloud(str(debug_dir / f"{stem}_objects.ply"), pcd_frame)

        if frame_idx % 20 == 0:
            print(f"[frame {frame_idx:05d}] {image_name} detections={len(detections)} object_rows={len(object_rows)}")

    if not all_points:
        raise RuntimeError("No object points were generated. Check bbox/depth pairing and score threshold.")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    raw_ply = out_dir / "objects_cloud_raw.ply"
    o3d.io.write_point_cloud(str(raw_ply), pcd)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size))

    out_ply = out_dir / "objects_cloud.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd)

    csv_path = out_dir / "objects_3d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(object_rows[0].keys()))
        writer.writeheader()
        writer.writerows(object_rows)

    centers_ply = None
    if args.save_object_centers_ply:
        centers_ply = out_dir / "object_centers.ply"
        save_object_centers_ply(object_rows, centers_ply)

    clusters_csv = None
    clusters_ply = None
    clusters = []
    if args.cluster_objects:
        clusters = cluster_object_centers(
            object_rows,
            cluster_radius_m=float(args.cluster_radius_m),
            min_cluster_size=int(args.cluster_min_count),
        )
        if clusters:
            clusters_csv = out_dir / "object_clusters.csv"
            clusters_ply = out_dir / "object_clusters.ply"
            save_clusters_csv(clusters, clusters_csv)
            save_clusters_ply(clusters, clusters_ply)

    print(f"[done] annotated frames: {ann_dir}")
    print(f"[done] debug per-frame object clouds: {debug_dir}")
    print(f"[done] raw object cloud: {raw_ply}")
    print(f"[done] voxel object cloud: {out_ply}")
    print(f"[done] object table: {csv_path}")
    if centers_ply is not None:
        print(f"[done] object centers: {centers_ply}")
    if clusters_csv is not None:
        print(f"[done] object clusters CSV: {clusters_csv}")
    if clusters_ply is not None:
        print(f"[done] object clusters PLY: {clusters_ply}")
    print(f"[info] points raw={len(points)}, voxel={len(pcd.points)}, detections={len(object_rows)}, clusters={len(clusters)}")

    if args.visualize:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
