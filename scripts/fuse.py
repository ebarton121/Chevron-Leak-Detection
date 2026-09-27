"""Build the synchronized LiDAR/thermal table for a recorded session.

  python scripts/fuse.py data/session_01 --config config/sensors.yaml
Writes data/session_01/fused.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.sync import fuse, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--config", default="config/sensors.yaml")
a = ap.parse_args()
s = Path(a.session)
out = fuse(s / "lidar_points.csv", s / "thermal", load_config(a.config))
out.to_csv(s / "fused.csv", index=False)
print(f"{len(out)} associated points across {out.frame_idx.nunique() if len(out) else 0} frames; "
      f"{int(out.in_image.sum()) if len(out) else 0} land inside the thermal image; "
      f"median |dt| {out.dt_ms.abs().median() if len(out) else float('nan'):.1f} ms")
