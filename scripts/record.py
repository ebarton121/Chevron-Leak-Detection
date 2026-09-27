"""Record LiDAR + thermal simultaneously on one host clock.

  python scripts/record.py --config config/sensors.yaml --out data/session_01
Stop with Ctrl+C. Produces lidar_points.csv, thermal/frames.csv + frames/*.png,
session_anchor.json.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync import lidar_ld, thermal  # noqa: E402
from leak_sync.clock import write_session_anchor  # noqa: E402
from leak_sync.sync import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sensors.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None, help="auto-stop")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    write_session_anchor(out)
    stop = threading.Event()
    counts, errors = {}, {}

    def run(name, fn):
        try:
            counts[name] = fn()
        except Exception as e:  # surface, don't hang the other thread
            errors[name] = e
            stop.set()

    lc, tc = cfg["lidar"], cfg["thermal"]
    threads = [
        threading.Thread(target=run, args=("lidar_points", lambda: lidar_ld.record(
            lc["port"], out / "lidar_points.csv", stop, lc.get("baud", 230400)))),
        threading.Thread(target=run, args=("thermal_frames", lambda: thermal.record(
            tc["device"], out / "thermal", stop, tc.get("raw16", False),
            tc.get("width"), tc.get("height"), tc.get("fps")))),
    ]
    for t in threads:
        t.start()
    t_end = time.monotonic() + a.seconds if a.seconds else None
    try:
        while any(t.is_alive() for t in threads) and not stop.is_set():
            if t_end and time.monotonic() > t_end:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=5)
    print("recorded:", counts)
    if errors:
        print("errors:", {k: repr(v) for k, v in errors.items()})
        sys.exit(1)


if __name__ == "__main__":
    main()
