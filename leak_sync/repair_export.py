"""Repair lidar_output_*.xlsx exports from the earlier capture script.

What is wrong with those files (verified on lidar_output_9_21_A1.xlsx):
  * Rows come in packets of (usually) 12 points. Each packet's FIRST angle is
    right (packet starts advance ~8-9 deg), but the angles inside a packet
    step 15-45 deg per point, so one packet claims to cover most of a circle
    while its 12 distances are nearly identical. Really they span ~8 deg.
  * ~3% of packets are short (rows lost in capture), so a fixed 12-row reshape
    drifts out of alignment. Packets are found by their constant angle step.
  * There are no timestamps.

The repair keeps each packet's start angle, re-spaces its points using the gap
to the next packet start, and rebuilds RELATIVE time from the rotation speed.
Absolute time is unknown; supply t0_ns if you have it from another source.
New recordings from lidar_ld.record() don't need any of this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PTS = 12


def find_packets(angles: np.ndarray, tol: float = 0.03) -> list[tuple[int, int]]:
    """Split rows into packets: consecutive rows whose angle step (mod 360)
    stays constant belong together, capped at 12 rows."""
    d = np.diff(angles) % 360.0
    pkts, s, n = [], 0, len(angles)
    while s < n:
        e = s + 1
        if e < n:
            step = d[s]
            e = s + 2
            while e < n and e - s < N_PTS and abs(d[e - 1] - step) < tol:
                e += 1
        pkts.append((s, e))
        s = e
    return pkts


def repair(df: pd.DataFrame, t0_ns: int = 0) -> tuple[pd.DataFrame, dict]:
    df = df.rename(columns={"speed_deg_sec": "speed_dps"}).reset_index(drop=True)
    ang = df["angle_deg"].to_numpy(dtype=float)
    speed = df["speed_dps"].to_numpy(dtype=float)
    pkts = find_packets(ang)
    starts = np.array([ang[s] for s, _ in pkts])
    sizes = np.array([e - s for s, e in pkts])

    gap = np.diff(starts) % 360.0
    typical_gap = float(np.median(gap[sizes[:-1] == N_PTS]))
    ok = (gap > 0.5 * typical_gap) & (gap < 1.5 * typical_gap)
    gap_fixed = np.where(ok, gap, typical_gap)
    pt_step = np.append(gap_fixed, typical_gap) / N_PTS
    unwrapped_start = np.concatenate([[0.0], np.cumsum(gap_fixed)])

    new_ang = np.empty_like(ang)
    unwrapped = np.empty_like(ang)
    pkt_id = np.empty(len(ang), dtype=int)
    for k, (s, e) in enumerate(pkts):
        i = np.arange(e - s)   # short packets: assume the tail rows were lost
        new_ang[s:e] = (starts[k] + pt_step[k] * i) % 360.0
        unwrapped[s:e] = unwrapped_start[k] + pt_step[k] * i
        pkt_id[s:e] = k

    t_s = unwrapped / speed
    t_s = np.maximum.accumulate(t_s)  # speed jitter must not reorder time
    rev = np.floor((unwrapped + starts[0]) / 360.0).astype(int)

    out = pd.DataFrame({
        "t_ns": (t0_ns + t_s * 1e9).astype(np.int64),
        "rev": rev,
        "packet": pkt_id,
        "angle_deg": new_ang.round(3),
        "distance_mm": df["distance_mm"].to_numpy(),
        "intensity": df["intensity"].to_numpy(),
        "speed_dps": speed.astype(int),
        "angle_deg_original": ang,
    })
    full = sizes == N_PTS
    orig_step = [np.median(np.diff(ang[s:e]) % 360) for (s, e), f in zip(pkts, full) if f]
    mean_speed = float(speed.mean())
    report = {
        "rows": int(len(df)),
        "packets": int(len(pkts)),
        "short_packets": int((~full).sum()),
        "rows_missing_est": int((N_PTS - sizes[~full]).sum()),
        "gaps_replaced": int((~ok).sum()),
        "orig_within_packet_step_median_deg": round(float(np.median(orig_step)), 2),
        "fixed_point_spacing_deg": round(typical_gap / N_PTS, 3),
        "mean_speed_dps": round(mean_speed, 1),
        "rotation_hz": round(mean_speed / 360.0, 2),
        "points_per_rev_est": int(round(360.0 / (typical_gap / N_PTS))),
        "revolutions": int(rev.max() - rev.min() + 1),
        "duration_s": round(float(t_s.max()), 2),
    }
    return out, report
