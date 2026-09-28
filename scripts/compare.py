"""Side-by-side comparison of two or more LiDAR recordings.

  python scripts/compare.py lidar_9_21_A1_repaired.csv lidar_9_21_A2_repaired.csv
  python scripts/compare.py data/session_01 data/session_02 --labels "Before" "After"
Writes compare.html (or --out PATH) and opens it. The 3D panels load three.js
from cdnjs, so they need an internet connection.
"""
import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leak_sync.analysis import flag_stray_points, summary  # noqa: E402
from leak_sync.sync import load_config  # noqa: E402


def load(src: Path) -> pd.DataFrame:
    if src.is_dir():
        src = src / "lidar_points.csv"
    return pd.read_csv(src)


def pack(df, stray, max_revs, max_pts):
    t0 = df.t_ns.min()
    df = df.assign(stray=stray)
    d = df[(df.distance_mm > 60) & (df.distance_mm < 12000)]
    revs = sorted(d.rev.unique())
    if len(revs) > max_revs:
        revs = [revs[i] for i in np.linspace(0, len(revs) - 1, max_revs).astype(int)]
    out = []
    for r in revs:
        g = d[d.rev == r]
        keep = g[g.stray | (np.arange(len(g)) % max(1, len(g) // max_pts) == 0)]
        out.append({"t": round(float((g.t_ns.median() - t0) / 1e9), 3),
                    "p": [[round(a, 1), int(x), int(s)] for a, x, s in zip(keep.angle_deg, keep.distance_mm, keep.stray)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--config", default=str(ROOT / "config" / "sensors.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-revs", type=int, default=150)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    labels = a.labels or [Path(s).stem.replace("_repaired", "") for s in a.srcs]
    sets = []
    for src, lab in zip(a.srcs, labels):
        df = load(Path(src))
        stray = flag_stray_points(df)
        sets.append({"label": lab, "stats": summary(df, stray), "revs": pack(df, stray, a.max_revs, 400)})
        print(lab, sets[-1]["stats"])
    payload = {"sets": sets, "hfov": float(np.degrees(np.arctan(cfg["cam_cx"] / cfg["cam_fx"])))}
    out = Path(a.out) if a.out else Path(a.srcs[0]).resolve().parent / "compare.html"
    html = (ROOT / "scripts" / "compare_template.html").read_text(encoding="utf-8")
    out.write_text(html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
