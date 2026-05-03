#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    fuse_depth_to_tsdf,
    sample_points_from_mesh,
)

def fuse_room_with_repo_utils(
    image_paths: list[str],
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    out_ply: str,
    voxel_length: float = 0.02,
    sdf_trunc: float = 0.08,
    max_depth: float = 10.0,
    num_points: int = 500000,
) -> None:
    """
    Fuse DA3 outputs into a point cloud using the official shared bench utilities.
    This is generic room fusion, not dataset-specific evaluation logic.
    """
    if len(image_paths) == 0:
        raise ValueError("No images to fuse.")
    if not (len(image_paths) == len(depths) == len(intrinsics) == len(extrinsics)):
        raise ValueError("Mismatched counts between images/depths/intrinsics/extrinsics.")

    images = []
    orig_sizes = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        orig_sizes.append((img.shape[0], img.shape[1]))

    images = np.stack(images, axis=0)

    model_h, model_w = depths.shape[1], depths.shape[2]

    depths_out = []
    intrinsics_out = []
    for i in range(len(depths)):
        orig_h, orig_w = orig_sizes[i]

        depth = cv2.resize(
            depths[i],
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
        depth[invalid_mask] = 0.0
        depths_out.append(depth)

        ixt = intrinsics[i].copy().astype(np.float32)
        ixt[0, :] *= orig_w / float(model_w)
        ixt[1, :] *= orig_h / float(model_h)
        intrinsics_out.append(ixt)

    depths_out = np.stack(depths_out, axis=0)
    intrinsics_out = np.stack(intrinsics_out, axis=0)

    volume = create_tsdf_volume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
    )

    mesh = fuse_depth_to_tsdf(
        volume=volume,
        depths=depths_out,
        images=images,
        intrinsics=intrinsics_out,
        extrinsics=extrinsics,
        max_depth=max_depth,
    )

    pcd = sample_points_from_mesh(mesh, num_points=num_points)

    ensure_dir(os.path.dirname(out_ply))
    o3d.io.write_point_cloud(out_ply, pcd)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sample_video_to_frames(
    video_path: str,
    out_dir: str,
    target_fps: float = 2.0,
    max_frames: int = 120,
    min_blur_var: float = 80.0,
    resize_long_edge: int | None = None,
) -> list[str]:
    """
    Extract frames from a video with light quality filtering.

    - samples approximately at target_fps
    - skips very blurry frames
    - optionally rescales long edge for storage/runtime

    Returns:
        sorted list of saved frame paths
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or np.isnan(src_fps):
        src_fps = 30.0

    stride = max(1, int(round(src_fps / target_fps)))

    saved = []
    frame_idx = 0
    kept = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < min_blur_var:
            frame_idx += 1
            continue

        if resize_long_edge is not None and resize_long_edge > 0:
            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if long_edge > resize_long_edge:
                scale = resize_long_edge / float(long_edge)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(out_dir, f"frame_{kept:05d}.jpg")
        cv2.imwrite(out_path, frame)
        saved.append(out_path)
        kept += 1

        if kept >= max_frames:
            break

        frame_idx += 1

    cap.release()
    return sorted(saved)


def collect_frames_from_dir(frame_dir: str, exts=(".jpg", ".jpeg", ".png")) -> list[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frame_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(frame_dir, f"*{ext.upper()}")))
    return sorted(paths)

def crop_image_by_fraction(
    image: np.ndarray,
    crop_left: float,
    crop_right: float,
    crop_top: float,
    crop_bottom: float,
) -> np.ndarray:
    """
    Crop image by fractional margins.
    Fractions are relative to width/height and must be in [0, 0.49].
    """
    h, w = image.shape[:2]

    if not (0.0 <= crop_left < 0.5 and 0.0 <= crop_right < 0.5 and
            0.0 <= crop_top < 0.5 and 0.0 <= crop_bottom < 0.5):
        raise ValueError("Crop fractions must be in [0, 0.5).")

    x1 = int(round(w * crop_left))
    x2 = int(round(w * (1.0 - crop_right)))
    y1 = int(round(h * crop_top))
    y2 = int(round(h * (1.0 - crop_bottom)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop values: empty crop.")

    return image[y1:y2, x1:x2]

def prepare_cropped_frames(
        image_paths: list[str],
        out_dir: str,
        crop_left: float,
        crop_right: float,
        crop_top: float,
        crop_bottom: float,
) -> list[str]:
    """
    Save cropped copies of input frames and return their paths.
    """
    ensure_dir(out_dir)
    out_paths = []

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        cropped = crop_image_by_fraction(
            img,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
        )

        out_path = os.path.join(out_dir, f"cropped_{i:05d}.png")
        cv2.imwrite(out_path, cropped)
        out_paths.append(out_path)

    return out_paths


def resize_depth_to_image(depth: np.ndarray, image_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_shape_hw
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def scale_intrinsics_to_image(ixt: np.ndarray, model_hw: tuple[int, int], image_hw: tuple[int, int]) -> np.ndarray:
    model_h, model_w = model_hw
    img_h, img_w = image_hw
    out = ixt.copy().astype(np.float32)
    out[0, :] *= img_w / float(model_w)
    out[1, :] *= img_h / float(model_h)
    return out


# def fuse_tsdf(
#     image_paths: list[str],
#     depths: np.ndarray,
#     intrinsics: np.ndarray,
#     extrinsics: np.ndarray,
#     out_ply: str,
#     voxel_length: float = 0.02,
#     sdf_trunc: float = 0.08,
#     max_depth: float = 10.0,
# ) -> None:
#     """
#     Fuse RGB + depth + camera poses into a TSDF and save as point cloud.
#     """
#     if len(image_paths) == 0:
#         raise ValueError("No images to fuse.")
#     if not (len(image_paths) == len(depths) == len(intrinsics) == len(extrinsics)):
#         raise ValueError("Mismatched counts between images/depths/intrinsics/extrinsics.")
#
#     volume = o3d.pipelines.integration.ScalableTSDFVolume(
#         voxel_length=voxel_length,
#         sdf_trunc=sdf_trunc,
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
#     )
#
#     model_h, model_w = depths.shape[1], depths.shape[2]
#
#     for idx, img_path in enumerate(image_paths):
#         bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         if bgr is None:
#             raise RuntimeError(f"Failed to read image: {img_path}")
#         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         img_h, img_w = rgb.shape[:2]
#
#         depth = resize_depth_to_image(depths[idx], (img_h, img_w))
#         depth[np.isnan(depth) | np.isinf(depth) | (depth <= 0)] = 0.0
#
#         ixt = scale_intrinsics_to_image(intrinsics[idx], (model_h, model_w), (img_h, img_w))
#         ext = extrinsics[idx].astype(np.float64)
#
#         depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
#         color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_o3d,
#             depth_o3d,
#             depth_scale=1.0,
#             depth_trunc=max_depth,
#             convert_rgb_to_intensity=False,
#         )
#
#         pinhole = o3d.camera.PinholeCameraIntrinsic(
#             img_w,
#             img_h,
#             float(ixt[0, 0]),
#             float(ixt[1, 1]),
#             float(ixt[0, 2]),
#             float(ixt[1, 2]),
#         )
#
#         volume.integrate(rgbd, pinhole, ext)
#
#     mesh = volume.extract_triangle_mesh()
#     mesh.compute_vertex_normals()
#
#     pcd = mesh.sample_points_uniformly(number_of_points=500000)
#     ensure_dir(os.path.dirname(out_ply))
#     o3d.io.write_point_cloud(out_ply, pcd)


def save_trajectory(extrinsics: np.ndarray, out_npy: str, out_txt: str) -> None:
    """
    Save world-to-camera extrinsics and camera centers in world coordinates.
    """
    ensure_dir(os.path.dirname(out_npy))
    np.save(out_npy, extrinsics)

    lines = []
    for i, ext in enumerate(extrinsics):
        R = ext[:3, :3]
        t = ext[:3, 3]
        cam_center_world = -R.T @ t
        lines.append(
            f"{i:05d} "
            f"{cam_center_world[0]:.6f} {cam_center_world[1]:.6f} {cam_center_world[2]:.6f}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_da3(
    image_paths: list[str],
    output_dir: str,
    model_name: str,
    process_res: int = 504,
    ref_view_strategy: str = "middle",
    use_ray_pose: bool = False,
    export_format: str = "mini_npz-glb",
    intrinsics: np.ndarray | None = None,
) -> dict:
    """
    Run official DA3 and return prediction object + extracted arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name).to(device=device)

    infer_kwargs = dict(
        image=image_paths,
        export_dir=output_dir,
        export_format=export_format,
        process_res=process_res,
        ref_view_strategy=ref_view_strategy,
        use_ray_pose=use_ray_pose,
        show_cameras=True,
    )

    if intrinsics is not None:
        infer_kwargs["intrinsics"] = intrinsics

    prediction = model.inference(**infer_kwargs)
    result = {
        "prediction": prediction,
        "depth": np.asarray(prediction.depth),
    }

    if hasattr(prediction, "extrinsics"):
        ext = np.asarray(prediction.extrinsics)
        if ext.shape[-2:] == (3, 4):
            # Convert to homogeneous 4x4 if needed
            ones = np.tile(np.array([0, 0, 0, 1], dtype=ext.dtype), (ext.shape[0], 1, 1))
            ext_h = np.concatenate([ext, np.zeros((ext.shape[0], 1, 4), dtype=ext.dtype)], axis=1)
            ext_h[:, 3, :] = np.array([0, 0, 0, 1], dtype=ext.dtype)
            ext = ext_h
        result["extrinsics"] = ext

    if hasattr(prediction, "intrinsics"):
        result["intrinsics"] = np.asarray(prediction.intrinsics)

    return result
def intrinsics_from_hfov_vfov(
    width: int,
    height: int,
    hfov_deg: float,
    vfov_deg: float,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """
    Build an approximate 3x3 pinhole intrinsics matrix from image size and
    horizontal/vertical field of view.

    Args:
        width: image width in pixels
        height: image height in pixels
        hfov_deg: horizontal field of view in degrees
        vfov_deg: vertical field of view in degrees
        cx: principal point x. Defaults to image center.
        cy: principal point y. Defaults to image center.

    Returns:
        K: 3x3 intrinsics matrix
    """
    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    fx = width / (2.0 * np.tan(hfov / 2.0))
    fy = height / (2.0 * np.tan(vfov / 2.0))

    if cx is None:
        cx = (width - 1) / 2.0
    if cy is None:
        cy = (height - 1) / 2.0

    """
    from config camera_info_example.yaml
    fx: 465.321303
    fy: 466.557484
    cx: 303.131049
    cy: 276.938277
    """

    fx = 465.321303
    fy = 466.557484
    cx = 303.131049
    cy = 276.938277


    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K

def cropped_fov_deg(
    hfov_deg: float,
    vfov_deg: float,
    crop_left: float,
    crop_right: float,
    crop_top: float,
    crop_bottom: float,
) -> tuple[float, float]:
    kept_w = 1.0 - crop_left - crop_right
    kept_h = 1.0 - crop_top - crop_bottom

    if kept_w <= 0 or kept_h <= 0:
        raise ValueError("Invalid crop fractions.")

    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    new_hfov = 2.0 * np.arctan(kept_w * np.tan(hfov / 2.0))
    new_vfov = 2.0 * np.arctan(kept_h * np.tan(vfov / 2.0))

    return float(np.rad2deg(new_hfov)), float(np.rad2deg(new_vfov))

def crop_intrinsics(
    K: np.ndarray,
    crop_x: int,
    crop_y: int,
) -> np.ndarray:
    """
    Shift principal point after cropping an image.
    crop_x, crop_y are the top-left crop offsets in original image coordinates.
    """
    K2 = K.copy().astype(np.float32)
    K2[0, 2] -= crop_x
    K2[1, 2] -= crop_y
    return K2

def extrinsics_to_camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    centers = []
    for ext in extrinsics:
        R = ext[:3, :3]
        t = ext[:3, 3]
        c = -R.T @ t
        centers.append(c)
    return np.asarray(centers, dtype=np.float64)


def make_trajectory_lineset(camera_centers_world: np.ndarray) -> o3d.geometry.LineSet:
    if len(camera_centers_world) < 2:
        raise ValueError("Need at least 2 camera centers for a trajectory line.")

    lines = [[i, i + 1] for i in range(len(camera_centers_world) - 1)]

    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(camera_centers_world)
    traj.lines = o3d.utility.Vector2iVector(lines)

    colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(lines), 1))
    traj.colors = o3d.utility.Vector3dVector(colors)
    return traj


def save_camera_centers_pcd(camera_centers_world: np.ndarray, out_path: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camera_centers_world)
    pcd.paint_uniform_color([0.0, 1.0, 0.0])
    o3d.io.write_point_cloud(out_path, pcd)


def export_per_frame_rgbd(
    image_paths: list[str],
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    out_dir: str,
    save_npz: bool = True,
    save_depth_png: bool = True,
    depth_vis_percentile: float = 98.0,
) -> None:
    """
    Save per-frame RGBD exports:
    - rgb_XXXXX.png
    - depth_XXXXX.npy
    - depth_vis_XXXXX.png
    - frame_XXXXX.npz  (rgb, depth, intrinsics, extrinsics)
    Depth is resized to the original RGB resolution, same as fusion path.
    """
    ensure_dir(out_dir)

    model_h, model_w = depths.shape[1], depths.shape[2]

    for i, img_path in enumerate(image_paths):
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        depth = cv2.resize(
            depths[i],
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
        depth[invalid_mask] = 0.0

        ixt = intrinsics[i].copy().astype(np.float32)
        ixt[0, :] *= orig_w / float(model_w)
        ixt[1, :] *= orig_h / float(model_h)

        stem = f"{i:05d}"

        # save rgb
        cv2.imwrite(
            os.path.join(out_dir, f"rgb_{stem}.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        # save raw depth
        np.save(os.path.join(out_dir, f"depth_{stem}.npy"), depth)

        # optional 16-bit depth png for easy archival/viewing
        if save_depth_png:
            valid = depth > 0
            depth_mm = np.zeros_like(depth, dtype=np.uint16)
            depth_mm[valid] = np.clip(depth[valid] * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(out_dir, f"depth_{stem}.png"), depth_mm)

        # save colored preview
        valid_vals = depth[depth > 0]
        if valid_vals.size > 0:
            vmax = np.percentile(valid_vals, depth_vis_percentile)
            vmax = max(vmax, 1e-6)
            depth_vis = np.clip(depth / vmax, 0.0, 1.0)
            depth_vis = (depth_vis * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            depth_vis[depth == 0] = 0
        else:
            depth_vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"depth_vis_{stem}.png"), depth_vis)

        # combined archive
        if save_npz:
            np.savez_compressed(
                os.path.join(out_dir, f"frame_{stem}.npz"),
                rgb=rgb,
                depth=depth,
                intrinsics=ixt,
                extrinsics=extrinsics[i],
                source_image_path=img_path,
            )


def visualize_map_and_trajectory(
    map_ply_path: str,
    trajectory_lines_path: str,
    camera_centers_ply_path: str | None = None,
) -> None:
    geoms = []

    room_map = o3d.io.read_point_cloud(map_ply_path)
    geoms.append(room_map)

    traj = o3d.io.read_line_set(trajectory_lines_path)
    geoms.append(traj)

    if camera_centers_ply_path is not None and os.path.exists(camera_centers_ply_path):
        cam_pts = o3d.io.read_point_cloud(camera_centers_ply_path)
        geoms.append(cam_pts)

    o3d.visualization.draw_geometries(geoms)


def main():
    parser = argparse.ArgumentParser(description="DA3-only room mapping/localization pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Input room video")
    src.add_argument("--frames_dir", type=str, help="Input folder of frames")

    parser.add_argument("--output_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--model", type=str, default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--target_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--min_blur_var", type=float, default=80.0)
    parser.add_argument("--resize_long_edge", type=int, default=1280)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--ref_view_strategy", type=str, default="middle")
    parser.add_argument("--use_ray_pose", action="store_true")
    parser.add_argument("--export_format", type=str, default="mini_npz-glb")
    parser.add_argument("--voxel_length", type=float, default=0.02)
    parser.add_argument("--sdf_trunc", type=float, default=0.08)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--crop_left", type=float, default=0.15)
    parser.add_argument("--crop_right", type=float, default=0.15)
    parser.add_argument("--crop_top", type=float, default=0.12)
    parser.add_argument("--crop_bottom", type=float, default=0.13)
    parser.add_argument("--disable_crop", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    frames_dir = os.path.join(output_dir, "frames")
    if args.video:
        image_paths = sample_video_to_frames(
            video_path=args.video,
            out_dir=frames_dir,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            min_blur_var=args.min_blur_var,
            resize_long_edge=args.resize_long_edge,
        )
    else:
        image_paths = collect_frames_from_dir(args.frames_dir)
        if len(image_paths) > args.max_frames > 0:
            step = max(1, len(image_paths) // args.max_frames)
            image_paths = image_paths[::step][:args.max_frames]


    if len(image_paths) < 2:
        raise RuntimeError("Need at least 2 frames.")

    print(f"[INFO] Using {len(image_paths)} frames")

    if args.disable_crop:
        working_image_paths = image_paths
        eff_hfov_deg = 130.0
        eff_vfov_deg = 90.0
    else:
        cropped_dir = os.path.join(output_dir, "cropped_frames")
        working_image_paths = prepare_cropped_frames(
            image_paths=image_paths,
            out_dir=cropped_dir,
            crop_left=args.crop_left,
            crop_right=args.crop_right,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
        )
        eff_hfov_deg, eff_vfov_deg = cropped_fov_deg(
            hfov_deg=130.0,
            vfov_deg=90.0,
            crop_left=args.crop_left,
            crop_right=args.crop_right,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
        )

    img0 = cv2.imread(working_image_paths[0], cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Failed to read image: {working_image_paths[0]}")

    h, w = img0.shape[:2]

    K_single = intrinsics_from_hfov_vfov(
        width=w,
        height=h,
        hfov_deg=eff_hfov_deg,
        vfov_deg=eff_vfov_deg,
    )

    manual_intrinsics = np.repeat(K_single[None, :, :], len(working_image_paths), axis=0)

    print("[INFO] Effective HFOV/VFOV after crop:", eff_hfov_deg, eff_vfov_deg)
    print("[INFO] Manual K:\n", K_single)

    da3_out_dir = os.path.join(output_dir, "da3_exports")
    ensure_dir(da3_out_dir)

    result = run_da3(
        image_paths=working_image_paths,
        output_dir=da3_out_dir,
        model_name=args.model,
        process_res=args.process_res,
        ref_view_strategy=args.ref_view_strategy,
        use_ray_pose=args.use_ray_pose,
        export_format=args.export_format,
        intrinsics=manual_intrinsics,
    )

    depth = result["depth"]
    if "extrinsics" not in result or "intrinsics" not in result:
        raise RuntimeError("DA3 output does not contain extrinsics/intrinsics, cannot build room map.")

    extrinsics = result["extrinsics"].astype(np.float32)
    intrinsics = result["intrinsics"].astype(np.float32)


    # Optional: also inspect what DA3 returned
    if "intrinsics" in result:
        pred_intrinsics = result["intrinsics"].astype(np.float32)
        print("[INFO] DA3 returned intrinsics[0]:\n", pred_intrinsics[0])
        print("[INFO] Manual intrinsics[0]:\n", manual_intrinsics[0])

        fx = intrinsics[0, 0, 0]
        fy = intrinsics[0, 1, 1]
        cx = intrinsics[0, 0, 2]
        cy = intrinsics[0, 1, 2]
        print(f"[INFO] Using downstream K[0]: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")


    np.save(os.path.join(output_dir, "depth.npy"), depth)
    np.save(os.path.join(output_dir, "intrinsics.npy"), intrinsics)
    np.save(os.path.join(output_dir, "extrinsics.npy"), extrinsics)

    camera_centers_world = extrinsics_to_camera_centers(extrinsics)

    traj_lines = make_trajectory_lineset(camera_centers_world)
    o3d.io.write_line_set(os.path.join(output_dir, "trajectory_lines.ply"), traj_lines)

    save_camera_centers_pcd(
        camera_centers_world,
        os.path.join(output_dir, "camera_centers.ply"),
    )

    rgbd_export_dir = os.path.join(output_dir, "rgbd_exports")
    export_per_frame_rgbd(
        image_paths=working_image_paths,
        depths=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        out_dir=rgbd_export_dir,
        save_npz=True,
        save_depth_png=True,
    )


    save_trajectory(
        extrinsics=extrinsics,
        out_npy=os.path.join(output_dir, "trajectory.npy"),
        out_txt=os.path.join(output_dir, "camera_centers_world.txt"),
    )

    fuse_room_with_repo_utils(
        image_paths=working_image_paths,
        depths=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        out_ply=os.path.join(output_dir, "room_map.ply"),
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        max_depth=args.max_depth,
    )

    visualize_map_and_trajectory(
        os.path.join(output_dir, "room_map.ply"),
        os.path.join(output_dir, "trajectory_lines.ply"),
        os.path.join(output_dir, "camera_centers.ply"),
    )

    print("[DONE] Outputs:")
    print(f"  - DA3 exports:   {da3_out_dir}")
    print(f"  - Depth:         {os.path.join(output_dir, 'depth.npy')}")
    print(f"  - Intrinsics:    {os.path.join(output_dir, 'intrinsics.npy')}")
    print(f"  - Extrinsics:    {os.path.join(output_dir, 'extrinsics.npy')}")
    print(f"  - Trajectory:    {os.path.join(output_dir, 'trajectory.npy')}")
    print(f"  - Cam centers:   {os.path.join(output_dir, 'camera_centers_world.txt')}")
    print(f"  - Room map:      {os.path.join(output_dir, 'room_map.ply')}")


if __name__ == "__main__":
    main()