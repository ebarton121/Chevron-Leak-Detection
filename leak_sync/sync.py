"""Associate LiDAR points with thermal frames on the shared clock, and
(optionally) project them into the thermal image to sample pixel values.

Output per associated point:
  frame_idx, t_frame_ns, t_ns, dt_ms, angle_deg, distance_mm, intensity,
  x_m, y_m, u, v, in_image, thermal_raw
`thermal_raw` is the camera's value at the pixel the LiDAR return lands on
(radiometric counts when recorded with raw16), i.e. a temperature profile along
the LiDAR scan line: the fused signal the leak model will consume.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def camera_model(cfg: dict):
    K = np.array([[cfg["cam_fx"], 0, cfg["cam_cx"]],
                  [0, cfg["cam_fy"], cfg["cam_cy"]],
                  [0, 0, 1]], dtype=float)
    dist = np.array([cfg.get(f"cam_d{i}", 0.0) for i in range(4)], dtype=float)
    R = np.array(cfg["extrin_calib"]["Rcl"], dtype=float).reshape(3, 3)
    t = np.array(cfg["extrin_calib"]["Pcl"], dtype=float)
    return K, dist, R, t


def lidar_xy(df: pd.DataFrame) -> np.ndarray:
    th = np.deg2rad(df["angle_deg"].to_numpy())
    r = df["distance_mm"].to_numpy() / 1000.0
    # LDRobot angles increase clockwise seen from above; flip to CCW (y left).
    return np.stack([r * np.cos(th), -r * np.sin(th)], axis=1)


def project(xy: np.ndarray, K, dist, R, t, w, h):
    pts = np.column_stack([xy, np.zeros(len(xy))])
    pc = pts @ R.T + t
    front = pc[:, 2] > 0.05
    uv = np.full((len(xy), 2), np.nan)
    if front.any():
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(pts[front], rvec, t, K, dist)
        uv[front] = proj.reshape(-1, 2)
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, inside


def associate(lidar: pd.DataFrame, frames: pd.DataFrame, max_dt_s: float,
              thermal_offset_s: float = 0.0) -> pd.DataFrame:
    """For every frame, take the LiDAR points within ±max_dt_s of its
    (offset-corrected) timestamp. Uses binary search, so it scales to hours."""
    lidar = lidar.sort_values("t_ns").reset_index(drop=True)
    t = lidar["t_ns"].to_numpy()
    win = int(max_dt_s * 1e9)
    parts = []
    for fr in frames.itertuples():
        tf = int(fr.t_ns + thermal_offset_s * 1e9)
        lo, hi = np.searchsorted(t, [tf - win, tf + win])
        if hi <= lo:
            continue
        seg = lidar.iloc[lo:hi].copy()
        seg.insert(0, "t_frame_ns", tf)
        seg.insert(0, "frame_idx", fr.frame_idx)
        seg["dt_ms"] = (seg["t_ns"] - tf) / 1e6
        parts.append(seg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fuse(lidar_csv: Path, thermal_dir: Path, cfg: dict, sample_pixels=True) -> pd.DataFrame:
    lidar = pd.read_csv(lidar_csv)
    rng = cfg.get("lidar", {})
    lidar = lidar[(lidar.distance_mm >= rng.get("min_range_mm", 20)) &
                  (lidar.distance_mm <= rng.get("max_range_mm", 12000))]
    frames = pd.read_csv(thermal_dir / "frames.csv")
    out = associate(lidar, frames, cfg["sync"]["max_dt_s"],
                    cfg["time_offset"]["thermal_time_offset_s"])
    if out.empty:
        return out

    xy = lidar_xy(out)
    out["x_m"], out["y_m"] = xy[:, 0], xy[:, 1]
    K, dist, R, t = camera_model(cfg)
    uv, inside = project(xy, K, dist, R, t, cfg["cam_width"], cfg["cam_height"])
    out["u"], out["v"], out["in_image"] = uv[:, 0], uv[:, 1], inside

    out["thermal_raw"] = np.nan
    if sample_pixels:
        files = frames.set_index("frame_idx")["file"]
        for fidx, grp in out[out.in_image].groupby("frame_idx"):
            img = cv2.imread(str(thermal_dir / files[fidx]), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ui = np.clip(grp.u.round().astype(int), 0, img.shape[1] - 1)
            vi = np.clip(grp.v.round().astype(int), 0, img.shape[0] - 1)
            out.loc[grp.index, "thermal_raw"] = img[vi, ui]
    return out


def estimate_time_offset(lidar: pd.DataFrame, frames_signal: pd.DataFrame,
                         sector_deg=(350, 10), search_s=0.5, step_s=0.005, center_s=0.0):
    """Estimate thermal_time_offset_s from a calibration recording.

    Procedure: wave a warm object (hand, mug of hot water) in and out of the
    LiDAR plane inside `sector_deg`, in front of the camera. The LiDAR sees
    range drop when the object enters; the camera's ROI mean rises at the same
    physical moment. `frames_signal` needs columns t_ns, value (ROI mean per
    frame). Returns the offset maximizing correlation of the two signals.
    """
    a0, a1 = sector_deg
    ang = lidar["angle_deg"]
    in_sec = (ang >= a0) | (ang <= a1) if a0 > a1 else (ang >= a0) & (ang <= a1)
    sec = lidar[in_sec].sort_values("t_ns")
    # One "pass" = one sweep of the beam through the sector (a sector that
    # straddles 0 deg spans two `rev` ids, so split on time gaps instead).
    gap = sec["t_ns"].diff().fillna(0) > 20e6
    sec = sec.assign(pass_id=gap.cumsum())
    # Closest return per pass, negated so "object present" = high.
    lid = sec.groupby("pass_id").agg(t_ns=("t_ns", "median"), value=("distance_mm", "min"))
    lid["value"] = -lid["value"]

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    ft = frames_signal["t_ns"].to_numpy() / 1e9
    fv = z(frames_signal["value"].to_numpy(dtype=float))
    lt = lid["t_ns"].to_numpy() / 1e9
    lv = z(lid["value"].to_numpy(dtype=float))
    best = (0.0, -np.inf)
    for off in center_s + np.arange(-search_s, search_s + step_s, step_s):
        interp = np.interp(ft + off, lt, lv, left=np.nan, right=np.nan)
        m = ~np.isnan(interp)
        if m.sum() < 10:
            continue
        c = np.corrcoef(interp[m], fv[m])[0, 1]
        if c > best[1]:
            best = (float(off), float(c))
    return {"thermal_time_offset_s": round(best[0], 4), "correlation": round(best[1], 3)}
