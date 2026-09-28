"""Run the real-time LiDAR + thermal pipeline.

Live hardware (ports/devices from config/sensors.yaml):
  python scripts/live.py --out data/live_01 --status-port 8765
Replay a recorded session as if it were live (no hardware needed):
  python scripts/live.py --replay data/session_01 --speed 2
Replay with injected sensor faults, in seconds of recording time:
  python scripts/live.py --replay data/session_01 --faults "lidar_gap:3-4,frozen:6-8,thermal_gap:9-10"
  (kinds: lidar_gap, thermal_gap, frozen, slow_spin)

Output: a status line per second on screen, and <out>/live.jsonl with every
fused-frame summary and health report. With --status-port, the latest state is
at http://localhost:<port>/status. Stop with Ctrl+C.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync.clock import write_session_anchor  # noqa: E402
from leak_sync.live import ConsoleSink, Faults, JsonlSink, LiveEngine  # noqa: E402
from leak_sync.sync import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "sensors.yaml"))
    ap.add_argument("--out", default="data/live")
    ap.add_argument("--replay", help="session folder to replay instead of live hardware")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--faults", help="replay fault injection, e.g. lidar_gap:3-4,frozen:6-8")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds")
    ap.add_argument("--status-port", type=int)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = Path(a.out)
    write_session_anchor(out)
    jsonl = JsonlSink(out / "live.jsonl")
    sinks = [jsonl] + ([] if a.quiet else [ConsoleSink()])
    eng = LiveEngine(cfg, sinks, replay=a.replay, speed=a.speed, faults=Faults.parse(a.faults),
                     status_port=a.status_port)
    print(f"Starting {'replay of ' + a.replay if a.replay else 'live capture'}; writing {out / 'live.jsonl'}"
          + (f"; status at http://localhost:{a.status_port}/status" if a.status_port else ""), flush=True)
    eng.start()
    t_end = time.monotonic() + a.seconds if a.seconds else None
    try:
        while True:
            time.sleep(0.2)
            if t_end and time.monotonic() > t_end:
                break
            if a.replay and eng.replay_done():
                time.sleep(0.3)  # let the last frames through
                break
    except KeyboardInterrupt:
        pass
    eng.stop()
    jsonl.close()
    print("Stopped.")


if __name__ == "__main__":
    main()
