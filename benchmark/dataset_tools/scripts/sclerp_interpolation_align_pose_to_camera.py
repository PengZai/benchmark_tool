#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path


from pytransform3d.transformations import (
    transform_from_pq,
    pq_from_transform,
    transform_sclerp,
)

def load_lidar_poses(path: str):
    """
    Load LiDAR poses from a text file with columns:
    timestamp x y z qx qy qz qw

    Returns:
      t: (N,) float64
      T: list of (4,4) transforms (world->lidar or whatever your trajectory represents)
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])

    if len(rows) < 2:
        raise ValueError("Need at least 2 pose samples.")

    data = np.asarray(rows, dtype=np.float64)
    t = data[:, 0]
    p = data[:, 1:4]
    q_xyzw = data[:, 4:8]  # (qx,qy,qz,qw) in your file

    # pytransform3d "pq" format is: (x, y, z, qw, qx, qy, qz)  :contentReference[oaicite:2]{index=2}
    pq = np.zeros((len(t), 7), dtype=np.float64)
    pq[:, 0:3] = p
    pq[:, 3] = q_xyzw[:, 3]      # qw
    pq[:, 4:7] = q_xyzw[:, 0:3]  # qx,qy,qz

    # Ensure sorted by time
    idx = np.argsort(t)
    t = t[idx]
    pq = pq[idx]

    # Build transforms
    T_list = [transform_from_pq(pq_i) for pq_i in pq]

    return t, T_list

def load_camera_timestamps(path: str):
    """
    Loads camera timestamps from a text file.
    Accepts:
      - one timestamp per line
      - or lines with multiple columns (timestamp in first column)
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name_of_image, str_timestamp = parts
            rows.append(float(str_timestamp))
    if not rows:
        raise ValueError("No camera timestamps found.")
    return np.asarray(rows, dtype=np.float64)

def interpolate_sclerp(t_lidar, T_lidar, t_query, clamp=True):
    """
    Piecewise ScLERP between the two nearest LiDAR poses.

    Args:
      clamp: if True, timestamps outside lidar range are clamped to endpoints.
             if False, they are skipped (return None).
    """
    t0 = t_lidar[0]
    tN = t_lidar[-1]

    if t_query <= t0:
        return T_lidar[0] if clamp else None
    if t_query >= tN:
        return T_lidar[-1] if clamp else None

    i = np.searchsorted(t_lidar, t_query) - 1
    i = int(np.clip(i, 0, len(t_lidar) - 2))

    ta, tb = t_lidar[i], t_lidar[i + 1]
    if tb <= ta:
        return T_lidar[i]  # degenerate, shouldn’t happen if times are strictly increasing

    u = (t_query - ta) / (tb - ta)
    # ScLERP for transformation matrices (SE(3))  :contentReference[oaicite:3]{index=3}
    return transform_sclerp(T_lidar[i], T_lidar[i + 1], float(u))

def write_output(path, t_cam, T_cam):
    """
    Writes:
    timestamp x y z qx qy qz qw
    """
    with open(path, "w") as f:
        f.write("#timestamp x y z q_x q_y q_z q_w\n")
        for ti, Ti in zip(t_cam, T_cam):
            pq = pq_from_transform(Ti)  # (x, y, z, qw, qx, qy, qz) :contentReference[oaicite:4]{index=4}
            x, y, z = pq[0:3]
            qw, qx, qy, qz = pq[3], pq[4], pq[5], pq[6]
            # back to your file order: qx qy qz qw
            f.write(f"{ti:.9f} {x:.12f} {y:.12f} {z:.12f} {qx:.12f} {qy:.12f} {qz:.12f} {qw:.12f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar_pose", default="/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/lio_sam_save_map/traj.txt", help="LiDAR pose trajectory file: timestamp x y z qx qy qz qw")
    ap.add_argument("--cam_ts", default="/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/time_colmap.txt", help="Camera timestamps file (timestamp per line or first column)")
    ap.add_argument("--output_name", default="traj_algin_with_colmap_timestamp.txt", help="Output pose file aligned to camera timestamps")
    ap.add_argument("--dt", type=float, default=0.0, help="Optional time offset: use lidar_pose(t_cam + dt). Units: seconds.")
    ap.add_argument("--no_clamp", action="store_true", help="If set, skip camera times outside LiDAR time range (instead of clamping).")
    args = ap.parse_args()

    t_lidar, T_lidar = load_lidar_poses(args.lidar_pose)
    t_cam = load_camera_timestamps(args.cam_ts)

    clamp = not args.no_clamp
    T_cam = []
    t_out = []

    for tc in t_cam:
        tq = tc + args.dt
        Ti = interpolate_sclerp(t_lidar, T_lidar, tq, clamp=clamp)
        if Ti is None:
            continue
        T_cam.append(Ti)
        t_out.append(tc)  # keep original camera timestamps in output

    if not T_cam:
        raise RuntimeError("No output poses produced (check timestamp ranges and --no_clamp).")

    path_lidar_pose = Path(args.lidar_pose)
    path_output = path_lidar_pose.with_name(args.output_name)
    write_output(path_output, np.asarray(t_out), T_cam)
    print(f"Wrote {len(T_cam)} poses to: {path_output}")

if __name__ == "__main__":
    main()