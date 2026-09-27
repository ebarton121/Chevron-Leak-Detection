"""Thermal camera capture over UVC (OpenCV).

Works for most USB thermal cameras that enumerate as a webcam. For radiometric
cameras (e.g. FLIR Lepton on a PureThermal board) use raw16=True to get the
16-bit Y16 counts instead of the colorized 8-bit preview; leak detection needs
the raw values.

Every frame is saved losslessly (PNG, 16-bit when raw16) and indexed in
frames.csv with a host monotonic timestamp. The timestamp is taken when the
frame is handed to us, so it lags true exposure; that lag is removed later
via `thermal_time_offset_s` in config/sensors.yaml (same idea as FAST-LIVO2's
img_time_offset).
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .clock import now_ns

FRAME_FIELDS = ["frame_idx", "t_ns", "file", "width", "height", "dtype"]


def open_camera(device: int | str, raw16: bool, width=None, height=None, fps=None):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open thermal camera {device!r}")
    if raw16:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def record(device, out_dir: Path, stop_event=None, raw16=False, width=None,
           height=None, fps=None, max_frames=None) -> int:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(device, raw16, width, height, fps)
    n = 0
    try:
        with open(out_dir / "frames.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
            w.writeheader()
            while not (stop_event is not None and stop_event.is_set()):
                ok, frame = cap.read()
                t = now_ns()
                if not ok:
                    continue
                if raw16 and frame.ndim == 2 and frame.dtype != "uint16":
                    frame = frame.view("uint16").reshape(frame.shape[0], -1)
                name = f"{n:07d}.png"
                cv2.imwrite(str(frames_dir / name), frame)
                w.writerow({"frame_idx": n, "t_ns": t, "file": f"frames/{name}",
                            "width": frame.shape[1], "height": frame.shape[0],
                            "dtype": str(frame.dtype)})
                n += 1
                if max_frames and n >= max_frames:
                    break
    finally:
        cap.release()
    return n
