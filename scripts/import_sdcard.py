"""Attach a Vue Pro R SD-card TIFF recording to a LiDAR session.

  python scripts/import_sdcard.py E:/DCIM/100_0001 data/session_01
Then: python scripts/estimate_offset.py data/session_01 --roi ... --search 30
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.thermal_sdcard import import_tiffs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tiff_dir")
ap.add_argument("session")
a = ap.parse_args()
rep = import_tiffs(Path(a.tiff_dir), Path(a.session))
print(json.dumps(rep, indent=2))
if "file_mtime" in rep["time_source"]:
    print("WARNING: some frames had no EXIF time; file times are usually only "
          "accurate to ~1-2 s. Rely on estimate_offset.py to align.")
