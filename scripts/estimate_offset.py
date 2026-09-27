"""Estimate thermal_time_offset_s from a 'wave a warm object' calibration session.

  python scripts/estimate_offset.py data/calib_01 --sector 350 10 --roi 60 40 40 40
ROI is x y w h in thermal pixels, covering where the object crosses the LiDAR plane.
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.sync import estimate_time_offset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--sector", nargs=2, type=float, default=[350, 10])
ap.add_argument("--roi", nargs=4, type=int, required=True)
ap.add_argument("--search", type=float, default=0.5, help="+/- seconds to search (use 30+ for SD-card TIFFs)")
a = ap.parse_args()
s = Path(a.session)
frames = pd.read_csv(s / "thermal" / "frames.csv")
x, y, w, h = a.roi
vals = []
for f in frames.file:
    img = cv2.imread(str(s / "thermal" / f), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals.append(float(img[y:y + h, x:x + w].mean()))
sig = pd.DataFrame({"t_ns": frames.t_ns, "value": vals})
lidar = pd.read_csv(s / "lidar_points.csv")
res = estimate_time_offset(lidar, sig, tuple(a.sector), search_s=a.search, step_s=0.005 if a.search <= 2 else 0.05)
if a.search > 2:  # coarse pass found the neighbourhood; refine around it
    res = estimate_time_offset(lidar, sig, tuple(a.sector), search_s=0.5, step_s=0.005,
                               center_s=res["thermal_time_offset_s"])
print(res)
print("Put thermal_time_offset_s into config/sensors.yaml (only trust it if correlation > ~0.6).")
