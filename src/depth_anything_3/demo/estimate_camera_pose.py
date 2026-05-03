#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

csv_path = "/home/daphnaa/Documents/xtend_da3_take_20260429_160647/estimated_camera_poses.csv"

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# 1. Filter for accepted frames only
# This is crucial: if we calculate distance between rejected frames,
# we get "teleportation" errors in our distance math.
if "accepted" in df.columns:
    df_accepted = df[df["accepted"] == True].copy()
else:
    df_accepted = df.copy()

# 2. Extract Coordinates
pts = df_accepted[["cam_x_right_m", "cam_y_down_m", "cam_z_forward_m"]].to_numpy()

# 3. Calculate Path Length (Sum of 3D segments)
# This measures the distance based on the smoothed positions you saved.
deltas = np.diff(pts, axis=0)
segment_lengths = np.linalg.norm(deltas, axis=1)
total_path_len = float(np.sum(segment_lengths))

# 4. Calculate Net Displacement
start_pos = pts[0]
end_pos = pts[-1]
net_displacement = float(np.linalg.norm(end_pos - start_pos))

# 5. Extract "Sensor distance" if available (The raw O3D output)
# This helps you see how much 'jitter' the smoothing filter removed.
raw_odom_dist = 0.0
if "odom_t_norm_m" in df_accepted.columns:
    raw_odom_dist = df_accepted["odom_t_norm_m"].sum()

print("-" * 40)
print(f"TRAJECTORY ANALYSIS: {os.path.basename(csv_path)}")
print("-" * 40)
print(f"Total Samples:       {len(df)}")
print(f"Accepted Updates:    {len(df_accepted)}")
print(f"Rejection Rate:      {((len(df)-len(df_accepted))/len(df))*100:.1f}%")
print("-" * 40)
print(f"Smoothed Path Length: {total_path_len:.3f} m")
if raw_odom_dist > 0:
    print(f"Raw Sensor Distance:  {raw_odom_dist:.3f} m (unfiltered sum)")
print(f"Net Displacement:     {net_displacement:.3f} m")
print("-" * 40)

print("Component Displacement (Start -> End):")
print(f"  Right (X):   {end_pos[0] - start_pos[0]: .3f} m")
print(f"  Down  (Y):   {end_pos[1] - start_pos[1]: .3f} m")
print(f"  Forward (Z): {end_pos[2] - start_pos[2]: .3f} m")

# Optional: Velocity analysis if you add the column later
if "velocity_m_s" in df_accepted.columns:
    avg_vel = df_accepted["velocity_m_s"].mean()
    max_vel = df_accepted["velocity_m_s"].max()
    print("-" * 40)
    print(f"Average Speed: {avg_vel:.2f} m/s")
    print(f"Max Speed:     {max_vel:.2f} m/s")