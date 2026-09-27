"""Import 14-bit TIFF recordings from a FLIR Vue Pro R SD card.

The camera saves TIFFs on its own clock, not the laptop's. This importer:
  1. reads each frame's capture time from EXIF (DateTimeOriginal + SubSecTime),
     falling back to the file's modified time if EXIF has no time;
  2. converts it onto the recording laptop's clock using session_anchor.json,
     assuming the camera clock is set to UTC (it only needs to be roughly right);
  3. writes thermal/frames.csv in the same format as live capture, so fuse.py
     and estimate_offset.py work unchanged.
Any camera-clock error is then measured with scripts/estimate_offset.py
(use --search to widen the window if the camera clock is off by seconds).

Pixel values are raw 14-bit sensor counts. They rise with temperature and are
fine for spotting anomalies; converting to deg C needs FLIR's calibration
metadata (FLIR Atlas SDK / FLIR Thermal Studio).
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .thermal import FRAME_FIELDS

EXIF_DATETIME_ORIGINAL, EXIF_SUBSEC_ORIGINAL, EXIF_DATETIME, EXIF_SUBSEC = 36867, 37521, 306, 37520


def frame_time_utc_ns(path: Path) -> tuple[int, str]:
    """Return (utc_ns, source) where source is 'exif' or 'file_mtime'.
    Looks in both the main TIFF directory and the EXIF sub-directory, since
    writers differ on where they put DateTime / SubSecTime."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            tags = dict(exif)
            try:
                tags.update(exif.get_ifd(0x8769))
            except Exception:
                pass
        dt = tags.get(EXIF_DATETIME_ORIGINAL) or tags.get(EXIF_DATETIME)
        if dt:
            ss = str(tags.get(EXIF_SUBSEC_ORIGINAL) or tags.get(EXIF_SUBSEC) or "0").strip().strip("\x00") or "0"
            t = datetime.strptime(str(dt).strip().strip("\x00"), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return int(t.timestamp() * 1e9 + float("0." + ss) * 1e9), "exif"
    except Exception:
        pass
    return int(path.stat().st_mtime_ns), "file_mtime"


def import_tiffs(src_dir: Path, session_dir: Path, copy=True) -> dict:
    files = sorted([p for p in Path(src_dir).iterdir() if p.suffix.lower() in (".tif", ".tiff")])
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files in {src_dir}")
    anchor = json.loads((Path(session_dir) / "session_anchor.json").read_text())
    utc_to_mono = anchor["monotonic_ns"] - anchor["utc_ns"]

    out = Path(session_dir) / "thermal"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows, sources = [], set()
    for p in files:
        utc, src = frame_time_utc_ns(p)
        sources.add(src)
        rows.append((utc, p))
    rows.sort(key=lambda r: r[0])

    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        w.writeheader()
        for i, (utc, p) in enumerate(rows):
            name = f"frames/{i:07d}{p.suffix.lower()}"
            if copy:
                shutil.copy2(p, out / name)
            with Image.open(p) as im:
                wdt, hgt, mode = im.width, im.height, im.mode
            w.writerow({"frame_idx": i, "t_ns": utc + utc_to_mono, "file": name,
                        "width": wdt, "height": hgt, "dtype": mode})
    dts = [(rows[i + 1][0] - rows[i][0]) / 1e9 for i in range(len(rows) - 1)]
    return {
        "frames": len(rows),
        "time_source": sorted(sources),
        "median_frame_interval_s": round(sorted(dts)[len(dts) // 2], 3) if dts else None,
        "span_s": round((rows[-1][0] - rows[0][0]) / 1e9, 2),
    }
