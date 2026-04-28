#!/usr/bin/env python3

import pandas as pd
import numpy as np

csv_path = "/home/user1/Documents/xtend_da3_takes/xtend_rectified_depth_da3_take_20260427_190026/estimated_camera_poses.csv"

df = pd.read_csv(csv_path)

# Keep only accepted pose updates if the column exists.
if "accepted" in df.columns:
    df = df[df["accepted"] == True].copy()

pts = df[["cam_x_right_m", "cam_y_down_m", "cam_z_forward_m"]].to_numpy()

deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
total_len = float(np.sum(deltas))

start = pts[0]
end = pts[-1]
net_displacement = float(np.linalg.norm(end - start))

print(f"Trajectory samples: {len(pts)}")
print(f"Total walked path length: {total_len:.3f} m")
print(f"Net displacement start->end: {net_displacement:.3f} m")

print()
print("Axis displacement:")
print(f"  right x:   {end[0] - start[0]: .3f} m")
print(f"  down y:    {end[1] - start[1]: .3f} m")
print(f"  forward z: {end[2] - start[2]: .3f} m")