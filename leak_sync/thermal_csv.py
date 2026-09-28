"""Import radiometric CSV exports (Scan_Number, Row_Index, Column_Index,
Temperature_Celsius) such as FLIR_Leak_Test.csv.

The CSV has no timestamps. To fuse with LiDAR, supply the scan interval and
the time of the first scan (to_session), then run estimate_offset.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_scans(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (temps[n_scans, H, W] in deg C as float32, scan_numbers)."""
    df = pd.read_csv(path, dtype={"Scan_Number": np.int32, "Row_Index": np.int32,
                                  "Column_Index": np.int32, "Temperature_Celsius": np.float32})
    scans = np.sort(df.Scan_Number.unique())
    h, w = int(df.Row_Index.max()) + 1, int(df.Column_Index.max()) + 1
    arr = np.full((len(scans), h, w), np.nan, np.float32)
    si = np.searchsorted(scans, df.Scan_Number.to_numpy())
    arr[si, df.Row_Index.to_numpy(), df.Column_Index.to_numpy()] = df.Temperature_Celsius.to_numpy()
    missing = int(np.isnan(arr).sum())
    if missing:
        raise ValueError(f"{path}: {missing} pixels missing (incomplete scans?)")
    return arr, scans


def to_session(arr: np.ndarray, scans: np.ndarray, session_dir: Path, t0_ns: int, interval_s: float):
    """Write scans as 16-bit PNGs in centi-kelvin (value/100 - 273.15 = deg C)
    plus thermal/frames.csv, so fuse.py / visualize.py / live replay can use them."""
    out = Path(session_dir) / "thermal"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "t_ns", "file", "width", "height", "dtype", "units"])
        w.writeheader()
        for i, (img, s) in enumerate(zip(arr, scans)):
            name = f"frames/{int(s):07d}.png"
            ck = np.clip(np.round((img + 273.15) * 100), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out / name), ck)
            w.writerow({"frame_idx": i, "t_ns": int(t0_ns + i * interval_s * 1e9), "file": name,
                        "width": img.shape[1], "height": img.shape[0], "dtype": "uint16", "units": "centikelvin"})
