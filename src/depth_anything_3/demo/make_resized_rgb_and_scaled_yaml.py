#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import yaml
import numpy as np


def load_matrix(data, key, shape):
    return np.array(data[key]["data"], dtype=float).reshape(shape)


def write_matrix(data, key, mat):
    data[key]["rows"] = int(mat.shape[0])
    data[key]["cols"] = int(mat.shape[1])
    data[key]["data"] = [float(x) for x in mat.reshape(-1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-camera-yaml", required=True)
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-width", type=int, default=504)
    p.add_argument("--out-height", type=int, default=392)
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_rgb_dir = out_dir / "rgb_504_392"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.raw_camera_yaml).expanduser(), "r") as f:
        cam = yaml.safe_load(f)

    raw_w = int(cam.get("image_width", cam.get("width")))
    raw_h = int(cam.get("image_height", cam.get("height")))

    sx = args.out_width / float(raw_w)
    sy = args.out_height / float(raw_h)

    cam["image_width"] = int(args.out_width)
    cam["image_height"] = int(args.out_height)

    if "camera_matrix" in cam:
        K = load_matrix(cam, "camera_matrix", (3, 3))
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        write_matrix(cam, "camera_matrix", K)

    if "projection_matrix" in cam:
        P = load_matrix(cam, "projection_matrix", (3, 4))
        P[0, 0] *= sx
        P[1, 1] *= sy
        P[0, 2] *= sx
        P[1, 2] *= sy
        P[0, 3] *= sx
        P[1, 3] *= sy
        write_matrix(cam, "projection_matrix", P)

    out_yaml = out_dir / "camera_xtend_ros_calib_504_392_full_resize.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cam, f, sort_keys=False)

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(rgb_dir.glob(ext))
    paths = sorted(set(paths))

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed to read {path}")
            continue
        resized = cv2.resize(img, (args.out_width, args.out_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_rgb_dir / path.name), resized)

    print(f"[done] wrote RGB:  {out_rgb_dir}")
    print(f"[done] wrote YAML: {out_yaml}")
    print(f"[info] scale sx={sx:.6f}, sy={sy:.6f}")


if __name__ == "__main__":
    main()