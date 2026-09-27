"""Fix angles and add relative timestamps to an old lidar_output_*.xlsx/csv.

  python scripts/repair_export.py lidar_output_9_21_A1.xlsx data/lidar_9_21_A1_repaired.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.repair_export import repair  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--t0-ns", type=int, default=0, help="absolute start time if known")
a = ap.parse_args()
src = Path(a.src)
df = pd.read_excel(src) if src.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(src)
out, report = repair(df, a.t0_ns)
Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.dst, index=False)
print(json.dumps(report, indent=2))
