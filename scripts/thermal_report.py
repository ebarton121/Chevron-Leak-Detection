"""Visual report for radiometric thermal scan CSVs (one or more, side by side).

  python scripts/thermal_report.py FLIR_Leak_Test.csv FLIR_Lidar_Aerosol_Spray.csv
Writes thermal_report.html (or --out PATH) and opens it.
"""
import argparse
import base64
import json
import sys
import warnings
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leak_sync.thermal_change import analyze  # noqa: E402
from leak_sync.thermal_csv import load_scans  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
LUT_INF = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
LUT_CW = (colormaps["coolwarm"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]


def jpg(bgr):
    ok, b = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def outline(img, mask, color):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, 2)


def build(path, delta_range):
    arr, scans = load_scans(path)
    rows, dc, cold, warm, onset = analyze(arr)
    lo, hi = (float(x) for x in np.percentile(arr, [0.5, 99.5]))
    frames = []
    for i in range(len(arr)):
        t = LUT_INF[np.clip((arr[i] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)]
        d = LUT_CW[np.clip((dc[i] / delta_range + 1) / 2 * 255, 0, 255).astype(np.uint8)]
        for img in (t, d):
            outline(img, cold[i], (255, 255, 0))
            outline(img, warm[i], (40, 40, 230))
        frames.append({"scan": int(scans[i]), "t": jpg(t), "d": jpg(d), **rows[i]})
    peak = max(rows, key=lambda r: r["cold_area_pct"])
    return {"name": Path(path).stem, "lo": round(lo, 1), "hi": round(hi, 1), "frames": frames,
            "onset_scan": int(scans[onset]) if onset is not None else None,
            "peak": {"scan": int(scans[peak["scan_index"]]), "area": peak["cold_area_pct"],
                     "drop": peak["cold_mean_drop_c"], "max_drop": min(r["cold_max_drop_c"] for r in rows)},
            "max_warm": max(r["warm_area_pct"] for r in rows),
            "recovered": rows[-1]["cold_area_pct"] < 0.5 * peak["cold_area_pct"],
            "last_area": rows[-1]["cold_area_pct"],
            "drift_total": rows[-1]["drift_c"], "size": [int(arr.shape[2]), int(arr.shape[1])],
            "no_base": rows[0]["no_baseline_pct"], "filled": rows[0]["baseline_from_later_frames_pct"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--delta-range", type=float, default=5.0, help="+/- deg C shown in the change map")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    sets = []
    for c in a.csvs:
        print("Analyzing", c, flush=True)
        sets.append(build(c, a.delta_range))
        s = sets[-1]
        print(f"  cold-change onset scan {s['onset_scan']}, peak {s['peak']}, warm max {s['max_warm']}%")
    out = Path(a.out) if a.out else Path(a.csvs[0]).resolve().parent / "thermal_report.html"
    html = (ROOT / "scripts" / "thermal_report_template.html").read_text(encoding="utf-8")
    out.write_text(html.replace("__PAYLOAD__", json.dumps({"sets": sets, "dr": a.delta_range})), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
