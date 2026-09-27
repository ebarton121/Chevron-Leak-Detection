"""Build an interactive HTML viewer for a recording. Opens in any browser.

  python scripts/visualize.py data/session_01            # LiDAR + thermal session
  python scripts/visualize.py lidar_9_21_A2_repaired.csv  # LiDAR-only CSV
Writes viewer.html next to the input (or --out PATH) and opens it.

Session mode shows, for each thermal frame: the LiDAR revolution closest in
time (top-down), the frame with the associated LiDAR returns projected onto it,
and the thermal value sampled at each return vs bearing. It runs the same
fuse() as scripts/fuse.py, so it uses config/sensors.yaml.
"""
import argparse
import base64
import json
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leak_sync.sync import fuse, load_config  # noqa: E402


def lidar_revs(df, max_revs, max_pts=450, max_range_mm=12000):
    df = df[(df.distance_mm > 20) & (df.distance_mm < max_range_mm)]
    revs = sorted(df.rev.unique())
    if len(revs) > max_revs:
        revs = [revs[i] for i in np.linspace(0, len(revs) - 1, max_revs).astype(int)]
    out = []
    for r in revs:
        g = df[df.rev == r]
        g = g.iloc[:: max(1, len(g) // max_pts)]
        out.append({"t": float(g.t_ns.median() / 1e9),
                    "p": [[round(a, 1), int(d)] for a, d in zip(g.angle_deg, g.distance_mm)]})
    return out


def frame_jpeg(path, lo, hi):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def build(src: Path, cfg_path: Path, max_frames: int, max_revs: int):
    cfg = load_config(cfg_path)
    payload = {"title": src.name, "frames": [], "revs": [], "w": cfg["cam_width"], "h": cfg["cam_height"],
               "hfov": float(np.degrees(np.arctan(cfg["cam_cx"] / cfg["cam_fx"])))}
    if src.is_dir():
        lidar = pd.read_csv(src / "lidar_points.csv")
        frames = pd.read_csv(src / "thermal" / "frames.csv")
        fused = fuse(src / "lidar_points.csv", src / "thermal", cfg)
        pick = frames.iloc[np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)]
        # one contrast range for the whole recording so brightness is comparable
        samp = [cv2.imread(str(src / "thermal" / f), cv2.IMREAD_UNCHANGED) for f in pick.file[:: max(1, len(pick) // 10)]]
        samp = np.concatenate([(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) if s.ndim == 3 else s).ravel() for s in samp if s is not None])
        lo, hi = np.percentile(samp, [1, 99])
        lidar_ok = lidar[(lidar.distance_mm > 20)]
        rev_t = lidar_ok.groupby("rev").t_ns.median()
        used_revs = set()
        for fr in pick.itertuples():
            img = frame_jpeg(src / "thermal" / fr.file, lo, hi)
            if img is None:
                continue
            fp = fused[(fused.frame_idx == fr.frame_idx) & fused.in_image] if len(fused) else fused
            rev = int((rev_t - fr.t_ns).abs().idxmin())
            used_revs.add(rev)
            payload["frames"].append({
                "t": fr.t_ns / 1e9, "img": img, "rev": rev,
                "pts": [[round(u, 1), round(v, 1), int(d), (None if pd.isna(tv) else float(tv)), round(a, 1)]
                        for u, v, d, tv, a in zip(fp.u, fp.v, fp.distance_mm, fp.thermal_raw, fp.angle_deg)]})
        payload["revs"] = {int(r): rv for r, rv in zip(
            sorted(used_revs), lidar_revs(lidar_ok[lidar_ok.rev.isin(used_revs)], len(used_revs) + 1))}
    else:
        df = pd.read_csv(src)
        payload["revs"] = dict(enumerate(lidar_revs(df, max_revs)))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="session folder or LiDAR CSV")
    ap.add_argument("--config", default=str(ROOT / "config" / "sensors.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--max-revs", type=int, default=150)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    payload = build(src, Path(a.config), a.max_frames, a.max_revs)
    out = Path(a.out) if a.out else (src if src.is_dir() else src.parent) / "viewer.html"
    html = (ROOT / "scripts" / "viewer_template.html").read_text(encoding="utf-8")
    out.write_text(html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
    n = len(payload["frames"]) or len(payload["revs"])
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} {'frames' if payload['frames'] else 'revolutions'})")
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
