#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def angle_diff_deg(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--min-yaw-step-deg", type=float, default=10.0)
    p.add_argument("--max-yaw-step-deg", type=float, default=25.0)
    args = p.parse_args()

    with open(args.in_json, "r") as f:
        data = json.load(f)

    kept = []
    last_yaw = None

    for item in data:
        yaw = float(item["pose"]["yaw"])

        if last_yaw is None:
            kept.append(item)
            last_yaw = yaw
            continue

        dy = abs(angle_diff_deg(last_yaw, yaw))

        if dy >= args.min_yaw_step_deg:
            if dy <= args.max_yaw_step_deg:
                kept.append(item)
                last_yaw = yaw
            else:
                print(f"[warn] large yaw jump skipped? dy={dy:.2f}, image={item.get('image')}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(kept, f, indent=2)

    print(f"[done] input frames: {len(data)}")
    print(f"[done] kept frames:  {len(kept)}")
    print(f"[done] wrote: {out}")


if __name__ == "__main__":
    main()