"""Real-time pipeline test: replays a synthetic session with injected faults.
Run: python tests/test_live.py"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.live import Faults, LiveEngine  # noqa: E402
from leak_sync.sync import load_config  # noqa: E402


class Collect:
    def __init__(self):
        self.fused, self.health = [], []

    def emit(self, kind, rec):
        (self.fused if kind == "fused" else self.health).append(rec)


def make_session(root: Path, seconds=8.0):
    (root / "thermal" / "frames").mkdir(parents=True)
    t = np.arange(0, seconds, 1 / 5000)
    ang = (t * 3575) % 360
    dist = np.full_like(t, 2300.0)                       # wall 2.3 m all around
    pd.DataFrame({"t_ns": (t * 1e9).astype(np.int64), "rev": (t * 9.93).astype(int), "angle_deg": ang,
                  "distance_mm": dist, "intensity": 200, "speed_dps": 3575}).to_csv(root / "lidar_points.csv", index=False)
    rows, rng = [], np.random.default_rng(0)
    yy, xx = np.mgrid[0:480, 0:640]
    for i, tf in enumerate(np.arange(0, seconds, 1 / 30)):
        img = 90 + rng.normal(0, 3, (480, 640)) + 120 * np.exp(-((xx - 420) ** 2) / 800 - ((yy - 250) ** 2) / 5000)
        cv2.imwrite(str(root / "thermal" / "frames" / f"{i:07d}.png"), np.clip(img, 0, 255).astype(np.uint8))
        rows.append({"frame_idx": i, "t_ns": int(tf * 1e9), "file": f"frames/{i:07d}.png"})
    pd.DataFrame(rows).to_csv(root / "thermal" / "frames.csv", index=False)


def test_live_replay_detects_faults():
    root = Path(tempfile.mkdtemp())
    make_session(root)
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "sensors.yaml")
    sink = Collect()
    eng = LiveEngine(cfg, [sink], replay=root, speed=2.0,
                     faults=Faults.parse("lidar_gap:2-3,frozen:4-6.5"))
    eng.start()
    while not eng.replay_done():
        time.sleep(0.1)
    time.sleep(0.3)
    eng.stop()

    ok_frames = [f for f in sink.fused if f["n_in_image"] > 0]
    assert len(sink.fused) > 150, len(sink.fused)
    # hot spot at u=420 -> bearing atan((420-320)/529) = 10.7 deg, on the 2.3 m wall
    b = np.median([f["max_bearing_deg"] for f in ok_frames])
    r = np.median([f["max_range_m"] for f in ok_frames])
    assert abs(b - 10.7) < 1.5 and abs(r - 2.3) < 0.05, (b, r)
    seen = {(k, c["status"]) for h in sink.health for k, c in h["checks"].items()}
    assert ("lidar_gaps", "WARN") in seen or ("lidar_data", "FAIL") in seen, seen
    assert ("thermal_frozen", "WARN") in seen, seen
    assert sink.health[-1]["checks"]["thermal_frozen"]["status"] == "OK"   # recovers
    lat = np.median([f["latency_ms"] for f in sink.fused])
    print(f"{len(sink.fused)} frames, hot spot {b:.1f} deg at {r:.2f} m, median latency {lat:.0f} ms, "
          f"faults seen: {sorted(k for k, s in seen if s != 'OK')}")


if __name__ == "__main__":
    test_live_replay_detects_faults()
    print("live test passed")
