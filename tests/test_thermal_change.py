"""Change detection on synthetic radiometric frames. Run: python tests/test_thermal_change.py"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.thermal_change import analyze  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def scene(n=30, cold_from=15, arm_until=6, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:240, 0:320]
    frames = []
    for i in range(n):
        f = 30 + 0.01 * yy + rng.normal(0, 0.1, yy.shape) - 0.05 * i          # scene + slow drift
        if i < arm_until:
            f[(xx < 80)] = 37.0                                                    # warm arm by the lens
        if i >= cold_from:                                                         # growing cold patch (spray)
            r = min(60, 16 + 8 * (i - cold_from))
            f[((xx - 220) ** 2 + (yy - 180) ** 2) < r * r] -= 3.0
        if i in (10, 11):                                                          # person walks through
            f[40:200, 140:170] = 36.5
        frames.append(f.astype(np.float32))
    return np.stack(frames)


def test_detects_spray_ignores_people_and_drift():
    rows, dc, cold, warm, onset = analyze(scene())
    assert onset == 15, onset                                  # spray found when it starts
    assert all(r["cold_area_pct"] == 0 for r in rows[:15])     # arm leaving and drift: no false cold
    assert rows[10]["warm_area_pct"] > 3                       # person flagged as warm, not cold
    assert abs(rows[-1]["cold_mean_drop_c"] + 3.0) < 0.3       # correct depth, drift removed
    cx, cy = rows[-1]["cold_centroid_px"]
    assert abs(cx - 220) < 5 and abs(cy - 180) < 8


def test_no_spray_no_onset():
    _, _, _, _, onset = analyze(scene(cold_from=999))
    assert onset is None


if __name__ == "__main__":
    test_detects_spray_ignores_people_and_drift(); test_no_spray_no_onset()
    print("thermal change tests passed")
