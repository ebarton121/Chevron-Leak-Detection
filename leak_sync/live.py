"""Real-time LiDAR + thermal fusion with sensor health monitoring.

Pipeline (one process, several threads):

    LiDAR source --> LidarBuffer (last few seconds, in memory) --+
                                                                 +--> Fuser --> sinks
    Thermal source --> frame queue ------------------------------+      |
                                                                        v
                                        HealthMonitor (every second) --> sinks

* Sources: live hardware (serial LiDAR, capture-dongle camera) or replay of a
  recorded session (for testing without hardware, with optional fault injection).
* Fuser: for each thermal frame, waits until the LiDAR points up to
  t_frame + max_dt have arrived (or a timeout), then associates, projects and
  samples thermal values, exactly as the offline fuse() does.
* HealthMonitor: flags lost sensors, low data rates, spin-rate problems,
  corrupted serial data, frozen video, fusion latency and clock drift.
* Sinks: anything with emit(kind, record). Built in: console, JSON-lines file,
  and an HTTP status endpoint. A Kafka/Xenese sink plugs in the same way.

All timestamps are host monotonic ns ("data time"). Replay shifts recorded
timestamps onto the current clock and paces delivery by --speed.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import pandas as pd

from .clock import DeviceClockMapper, now_ns
from .sync import camera_model, project

# ----------------------------------------------------------------------------
# Shared LiDAR buffer
# ----------------------------------------------------------------------------


class LidarBuffer:
    """Thread-safe store of the last `keep_s` seconds of LiDAR points.
    Points arrive in small batches (numpy structured chunks)."""

    COLS = ("t_ns", "angle_deg", "distance_mm", "intensity", "speed_dps")

    def __init__(self, keep_s: float = 3.0):
        self.keep_ns = int(keep_s * 1e9)
        self._chunks: deque = deque()
        self._lock = threading.Lock()
        self.latest_t = -1
        self.last_add_wall_ns = None
        self.cond = threading.Condition(self._lock)

    def add(self, chunk: dict):
        if len(chunk["t_ns"]) == 0:
            return
        with self.cond:
            self._chunks.append(chunk)
            self.latest_t = max(self.latest_t, int(chunk["t_ns"][-1]))
            self.last_add_wall_ns = now_ns()
            cut = self.latest_t - self.keep_ns
            while self._chunks and self._chunks[0]["t_ns"][-1] < cut:
                self._chunks.popleft()
            self.cond.notify_all()

    def wait_until(self, t_ns: int, timeout_s: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.latest_t >= t_ns, timeout=timeout_s)

    def query(self, t0: int, t1: int) -> dict:
        with self._lock:
            parts = [c for c in self._chunks if c["t_ns"][-1] >= t0 and c["t_ns"][0] <= t1]
        if not parts:
            return {k: np.array([]) for k in self.COLS}
        cat = {k: np.concatenate([p[k] for p in parts]) for k in self.COLS}
        m = (cat["t_ns"] >= t0) & (cat["t_ns"] <= t1)
        return {k: v[m] for k, v in cat.items()}


@dataclass
class Frame:
    idx: int
    t_ns: int          # data time of the frame
    arrival_ns: int    # host time we received it (for latency)
    image: np.ndarray


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------


class SourceBase(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name, daemon=True)
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self.last_arrival_ns: Optional[int] = None
        self.count = 0

    def stop(self):
        self.stop_event.set()


class LidarSerialSource(SourceBase):
    def __init__(self, port: str, buf: LidarBuffer, baud: int = 230400, batch_ms: int = 20):
        super().__init__("lidar")
        self.port, self.buf, self.baud, self.batch_ns = port, buf, baud, batch_ms * 1_000_000
        self.stats: dict = {}
        self.mapper = DeviceClockMapper(wrap=30000, units_per_s=1000.0)

    def run(self):
        from . import lidar_ld
        try:
            import serial
            with serial.Serial(self.port, self.baud, timeout=0.1) as ser:
                def read(k):
                    return None if self.stop_event.is_set() else ser.read(k)
                pending = {c: [] for c in LidarBuffer.COLS}
                t_flush = now_ns()
                for pt in lidar_ld.packets_to_points(lidar_ld.iter_packets(read, self.stats), self.mapper):
                    for c in LidarBuffer.COLS:
                        pending[c].append(pt[c])
                    self.count += 1
                    self.last_arrival_ns = now_ns()
                    if self.last_arrival_ns - t_flush >= self.batch_ns:
                        self.buf.add({c: np.asarray(v) for c, v in pending.items()})
                        pending = {c: [] for c in LidarBuffer.COLS}
                        t_flush = self.last_arrival_ns
        except Exception as e:  # surfaced by the health monitor
            self.error = f"{type(e).__name__}: {e}"


class CameraSource(SourceBase):
    def __init__(self, device, out_q: "queue.Queue[Frame]", raw16=False, width=None, height=None, fps=None):
        super().__init__("thermal")
        self.args = (device, raw16, width, height, fps)
        self.q = out_q
        self.dropped = 0

    def run(self):
        from .thermal import open_camera
        try:
            cap = open_camera(*self.args)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            return
        try:
            while not self.stop_event.is_set():
                ok, img = cap.read()
                t = now_ns()
                if not ok:
                    time.sleep(0.01)
                    continue
                self.last_arrival_ns = t
                _put_latest(self, Frame(self.count, t, t, img))
                self.count += 1
        finally:
            cap.release()


def _put_latest(src, frame):
    """Never block the camera: if the fuser is behind, drop the oldest frame."""
    try:
        src.q.put_nowait(frame)
    except queue.Full:
        try:
            src.q.get_nowait()
            src.dropped += 1
        except queue.Empty:
            pass
        src.q.put_nowait(frame)


@dataclass
class Faults:
    """Replay fault injection, in seconds of recording time: [(start, end), ...]."""
    lidar_gap: List[tuple] = field(default_factory=list)
    thermal_gap: List[tuple] = field(default_factory=list)
    frozen: List[tuple] = field(default_factory=list)
    slow_spin: List[tuple] = field(default_factory=list)

    @staticmethod
    def parse(spec: Optional[str]) -> "Faults":
        f = Faults()
        for item in filter(None, (spec or "").split(",")):
            kind, rng = item.split(":")
            a, b = (float(x) for x in rng.split("-"))
            getattr(f, kind).append((a, b))
        return f

    @staticmethod
    def active(windows, t_s):
        return any(a <= t_s < b for a, b in windows)


class ReplayClock:
    """Maps recording time onto 'now' and paces delivery by `speed`."""

    def __init__(self, t0_rec_ns: int, speed: float):
        self.t0_rec, self.speed = t0_rec_ns, speed
        self.t0_live = now_ns()

    def to_live(self, t_rec_ns):
        return self.t0_live + (np.asarray(t_rec_ns) - self.t0_rec)

    def rec_s(self, t_rec_ns):
        return (t_rec_ns - self.t0_rec) / 1e9

    def sleep_until(self, t_rec_ns, stop_event):
        target = self.t0_live + (t_rec_ns - self.t0_rec) / self.speed
        while not stop_event.is_set():
            d = (target - now_ns()) / 1e9
            if d <= 0:
                return
            time.sleep(min(d, 0.05))


class LidarReplaySource(SourceBase):
    def __init__(self, csv_path: Path, buf: LidarBuffer, clock: ReplayClock, faults: Faults, batch_ms=20):
        super().__init__("lidar")
        df = pd.read_csv(csv_path)
        if "speed_dps" not in df and "speed_deg_sec" in df:
            df = df.rename(columns={"speed_deg_sec": "speed_dps"})
        self.df = df.sort_values("t_ns").reset_index(drop=True)
        self.buf, self.clock, self.faults = buf, clock, faults
        self.batch_ns = batch_ms * 1_000_000
        self.stats = {"packets": 0, "bad_packets": 0}
        self.mapper = None

    def run(self):
        t = self.df.t_ns.to_numpy()
        edges = np.arange(t[0], t[-1] + self.batch_ns, self.batch_ns)
        idx = np.searchsorted(t, edges)
        for i in range(len(idx) - 1):
            if self.stop_event.is_set():
                return
            lo, hi = idx[i], idx[i + 1]
            if hi <= lo:
                continue
            self.clock.sleep_until(int(t[hi - 1]), self.stop_event)
            rs = self.clock.rec_s(t[lo])
            if Faults.active(self.faults.lidar_gap, rs):
                continue
            seg = self.df.iloc[lo:hi]
            speed = seg.speed_dps.to_numpy(dtype=float)
            if Faults.active(self.faults.slow_spin, rs):
                speed = speed * 0.6
            self.buf.add({"t_ns": self.clock.to_live(seg.t_ns.to_numpy()).astype(np.int64),
                          "angle_deg": seg.angle_deg.to_numpy(float),
                          "distance_mm": seg.distance_mm.to_numpy(float),
                          "intensity": seg.intensity.to_numpy(float) if "intensity" in seg else np.zeros(len(seg)),
                          "speed_dps": speed})
            self.stats["packets"] += max(1, (hi - lo) // 12)
            self.count += hi - lo
            self.last_arrival_ns = now_ns()


class ThermalReplaySource(SourceBase):
    def __init__(self, thermal_dir: Path, out_q, clock: ReplayClock, faults: Faults):
        super().__init__("thermal")
        self.dir = Path(thermal_dir)
        self.frames = pd.read_csv(self.dir / "frames.csv").sort_values("t_ns")
        self.q, self.clock, self.faults = out_q, clock, faults
        self.dropped = 0

    def run(self):
        frozen_img = None
        for fr in self.frames.itertuples():
            if self.stop_event.is_set():
                return
            self.clock.sleep_until(int(fr.t_ns), self.stop_event)
            rs = self.clock.rec_s(fr.t_ns)
            if Faults.active(self.faults.thermal_gap, rs):
                continue
            img = cv2.imread(str(self.dir / fr.file), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if Faults.active(self.faults.frozen, rs):
                frozen_img = img if frozen_img is None else frozen_img
                img = frozen_img
            else:
                frozen_img = None
            t_live = int(self.clock.to_live(fr.t_ns))
            self.last_arrival_ns = now_ns()
            _put_latest(self, Frame(self.count, t_live, self.last_arrival_ns, img))
            self.count += 1


# ----------------------------------------------------------------------------
# Fuser
# ----------------------------------------------------------------------------


class Fuser(threading.Thread):
    def __init__(self, cfg: dict, buf: LidarBuffer, in_q, emit: Callable, speed: float = 1.0,
                 max_wait_s: float = 0.15):
        super().__init__(name="fuser", daemon=True)
        self.cfg, self.buf, self.q, self.emit = cfg, buf, in_q, emit
        self.max_dt_ns = int(cfg["sync"]["max_dt_s"] * 1e9)
        self.offset_ns = int(cfg["time_offset"]["thermal_time_offset_s"] * 1e9)
        self.max_wait_s = max_wait_s / speed
        self.speed = speed
        self.cam = camera_model(cfg)
        rng = cfg.get("lidar", {})
        self.rmin, self.rmax = rng.get("min_range_mm", 20), rng.get("max_range_mm", 12000)
        self.stop_event = threading.Event()
        self.latencies: deque = deque(maxlen=300)
        self.results: deque = deque(maxlen=300)     # (t_ns, n_points, incomplete)
        self.last_diff: deque = deque(maxlen=300)   # (t_ns, mean abs diff vs previous frame)
        self._prev = None
        self.latest = None

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            try:
                fr: Frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.emit("fused", self.process(fr))

    def process(self, fr: Frame) -> dict:
        tf = fr.t_ns + self.offset_ns
        # Wait for the LiDAR to catch up to tf + max_dt, but only if it is alive
        # (within 0.5 s of this frame); otherwise fuse immediately with what we have.
        la = self.buf.last_add_wall_ns
        lidar_alive = la is not None and (now_ns() - la) / 1e9 < 0.2 / self.speed
        complete = self.buf.wait_until(tf + self.max_dt_ns, self.max_wait_s if lidar_alive else 0.0)
        pts = self.buf.query(tf - self.max_dt_ns, tf + self.max_dt_ns)
        m = (pts["distance_mm"] >= self.rmin) & (pts["distance_mm"] <= self.rmax)
        pts = {k: v[m] for k, v in pts.items()}

        img = fr.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self._prev is not None and self._prev.shape == gray.shape:
            self.last_diff.append((fr.t_ns, float(np.mean(np.abs(gray.astype(np.float32) - self._prev)))))
        self._prev = gray.astype(np.float32)

        n = len(pts["t_ns"])
        rec = {"frame_idx": fr.idx, "t_frame_ns": tf, "n_points": n, "incomplete": not complete,
               "n_in_image": 0, "max_value": None, "max_bearing_deg": None, "max_range_m": None}
        if n:
            th = np.deg2rad(pts["angle_deg"])
            r = pts["distance_mm"] / 1000.0
            xy = np.stack([r * np.cos(th), -r * np.sin(th)], axis=1)
            K, dist, R, t = self.cam
            uv, inside = project(xy, K, dist, R, t, gray.shape[1], gray.shape[0])
            if inside.any():
                ui = np.clip(uv[inside, 0].round().astype(int), 0, gray.shape[1] - 1)
                vi = np.clip(uv[inside, 1].round().astype(int), 0, gray.shape[0] - 1)
                vals = gray[vi, ui].astype(float)
                j = int(np.argmax(vals))
                b = pts["angle_deg"][inside][j]
                rec.update(n_in_image=int(inside.sum()), max_value=float(vals[j]),
                           max_bearing_deg=round(float(b - 360 if b > 180 else b), 2),
                           max_range_m=round(float(r[inside][j]), 3))
                self.latest = {"frame": fr, "u": uv[inside, 0], "v": uv[inside, 1], "vals": vals,
                               "angle": pts["angle_deg"][inside], "range_m": r[inside]}
        lat = (now_ns() - fr.arrival_ns) / 1e6
        rec["latency_ms"] = round(lat, 1)
        self.latencies.append(lat)
        self.results.append((fr.t_ns, rec["n_in_image"], rec["incomplete"]))
        return rec


# ----------------------------------------------------------------------------
# Health monitoring
# ----------------------------------------------------------------------------

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_RANK = {OK: 0, WARN: 1, FAIL: 2}


class HealthMonitor(threading.Thread):
    """Evaluates sensor and pipeline health every `period_s` over the last `window_s`."""

    def __init__(self, cfg, buf, lidar_src, cam_src, fuser, emit, speed=1.0, period_s=1.0, window_s=2.0):
        super().__init__(name="health", daemon=True)
        self.cfg, self.buf, self.ls, self.cs, self.fu, self.emit = cfg, buf, lidar_src, cam_src, fuser, emit
        self.speed, self.period, self.window_ns = speed, period_s / speed, int(window_s * 1e9)
        h = cfg.get("health", {})
        self.lim = {
            "lidar_min_pts_s": h.get("lidar_min_points_per_s", 2500),
            "spin_hz": tuple(h.get("spin_hz_range", [8.0, 12.0])),
            "bad_pkt_pct": h.get("max_bad_packet_pct", 2.0),
            "silence_s": h.get("sensor_silence_s", 0.5),
            "thermal_min_fps": h.get("thermal_min_fps", 0.5 * (cfg.get("thermal", {}).get("fps") or 30)),
            "frozen_s": h.get("frozen_s", 2.0),
            "frozen_diff": h.get("frozen_mean_abs_diff", 0.05),
            "latency_ms": h.get("max_latency_ms_p95", 500),
            "min_assoc_frac": h.get("min_associated_frame_frac", 0.5),
            "drift_ms_per_min": h.get("max_clock_drift_ms_per_min", 20),
        }
        self.stop_event = threading.Event()
        self.state = {}
        self._pkts = deque(maxlen=120)
        self._offsets = deque(maxlen=600)
        self._drops = deque()
        self.latest = None

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.wait(self.period):
            self.emit("health", self.check())

    def _silence(self, src):
        if src.last_arrival_ns is None:
            return None
        return (now_ns() - src.last_arrival_ns) / 1e9 * self.speed

    def check(self) -> dict:
        L, checks = self.lim, {}
        t_now = max(self.buf.latest_t, self.fu.results[-1][0] if self.fu.results else -1)

        def put(name, status, value, detail=""):
            checks[name] = {"status": status, "value": value, "detail": detail}

        # --- LiDAR ---
        if self.ls.error:
            put("lidar_connection", FAIL, None, self.ls.error)
        s = self._silence(self.ls)
        if s is None:
            put("lidar_data", FAIL if self.ls.error else WARN, None, "no LiDAR data yet")
        else:
            put("lidar_data", FAIL if s > L["silence_s"] else OK, round(s, 2), "seconds since last LiDAR data")
            win = self.buf.query(t_now - self.window_ns, t_now)
            span = (win["t_ns"][-1] - win["t_ns"][0]) / 1e9 if len(win["t_ns"]) > 1 else 0
            rate = len(win["t_ns"]) / max(span, 1e-3) if span > 0.2 else 0.0
            if s <= L["silence_s"]:
                put("lidar_rate", OK if rate >= L["lidar_min_pts_s"] else WARN, round(rate), "points per second")
            if len(win["t_ns"]) > 1:
                gap = float(np.max(np.diff(win["t_ns"]))) / 1e9
                put("lidar_gaps", OK if gap <= 0.3 else WARN, round(gap, 2), "longest gap in LiDAR data, last 2 s")
            recent_spin = win["speed_dps"][win["t_ns"] >= t_now - 500_000_000] if len(win["t_ns"]) else win["speed_dps"]
            if len(recent_spin):
                hz = float(np.median(recent_spin)) / 360
                lo, hi = L["spin_hz"]
                put("spin_rate", OK if lo <= hz <= hi else WARN, round(hz, 2), "revolutions per second")
            st = getattr(self.ls, "stats", {})
            self._pkts.append((time.monotonic(), st.get("packets", 0), st.get("bad_packets", 0)))
            old = self._pkts[0]
            dp, db = st.get("packets", 0) - old[1], st.get("bad_packets", 0) - old[2]
            pct = 100 * db / max(dp + db, 1)
            put("serial_integrity", OK if pct <= L["bad_pkt_pct"] else WARN, round(pct, 2), "% corrupted packets")
            mapper = getattr(self.ls, "mapper", None)
            if mapper is not None and mapper.offset_ns is not None:
                self._offsets.append((time.monotonic(), mapper.offset_ns))
                if self._offsets[-1][0] - self._offsets[0][0] > 10:
                    (ta, oa), (tb, ob) = self._offsets[0], self._offsets[-1]
                    drift = abs(ob - oa) / 1e6 / ((tb - ta) / 60)
                    put("clock_drift", OK if drift <= L["drift_ms_per_min"] else WARN, round(drift, 2),
                        "ms per minute, LiDAR clock vs host")

        # --- Thermal ---
        if self.cs.error:
            put("thermal_connection", FAIL, None, self.cs.error)
        s = self._silence(self.cs)
        if s is None:
            put("thermal_data", FAIL if self.cs.error else WARN, None, "no thermal frames yet")
        else:
            put("thermal_data", FAIL if s > max(L["silence_s"], 3 / max(L["thermal_min_fps"], 0.1)) else OK,
                round(s, 2), "seconds since last frame")
            recent = [t for t, _, _ in self.fu.results if t >= t_now - self.window_ns]
            fps = (len(recent) - 1) / ((recent[-1] - recent[0]) / 1e9) if len(recent) > 2 else 0.0
            put("thermal_fps", OK if fps >= L["thermal_min_fps"] else WARN, round(fps, 1), "frames per second")
            if len(recent) > 1:
                fgap = max(b - a for a, b in zip(recent, recent[1:])) / 1e9
                lim = max(0.3, 3 / max(L["thermal_min_fps"], 0.1))
                put("thermal_gaps", OK if fgap <= lim else WARN, round(fgap, 2), "longest gap between frames, last 2 s")
            diffs = [(t, d) for t, d in self.fu.last_diff if t >= t_now - int(L["frozen_s"] * 1e9)]
            if len(diffs) >= 3:
                frozen = all(d < L["frozen_diff"] for _, d in diffs)
                put("thermal_frozen", WARN if frozen else OK, round(max(d for _, d in diffs), 3),
                    "max frame-to-frame change (0 = identical frames)")
            dropped_total = getattr(self.cs, "dropped", 0)
            self._drops.append((time.monotonic(), dropped_total))
            while len(self._drops) > 1 and self._drops[0][0] < time.monotonic() - self.window_ns / 1e9 / self.speed:
                self._drops.popleft()
            dropped = dropped_total - self._drops[0][1]
            put("frames_dropped", OK if dropped == 0 else WARN, dropped, "frames skipped recently because fusion fell behind")

        # --- Fusion ---
        if self.fu.latencies:
            p95 = float(np.percentile(list(self.fu.latencies)[-100:], 95))
            put("fusion_latency", OK if p95 <= L["latency_ms"] else WARN, round(p95, 1), "ms, p95 frame-in to result-out")
        res = [r for r in self.fu.results if r[0] >= t_now - self.window_ns]
        if res:
            frac = sum(1 for _, n, _ in res if n > 0) / len(res)
            put("association", OK if frac >= L["min_assoc_frac"] else WARN, round(frac, 2),
                "fraction of frames with LiDAR returns in the image")

        overall = max((c["status"] for c in checks.values()), key=_RANK.get, default=WARN)
        changes = {k: c["status"] for k, c in checks.items() if self.state.get(k) != c["status"]}
        self.state.update({k: c["status"] for k, c in checks.items()})
        rec = {"t_ns": t_now, "overall": overall, "checks": checks, "changed": changes}
        self.latest = rec
        return rec


# ----------------------------------------------------------------------------
# Sinks
# ----------------------------------------------------------------------------


class JsonlSink:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def emit(self, kind, rec):
        with self.lock:
            self.f.write(json.dumps({"kind": kind, "wall_utc_ns": time.time_ns(), **rec}) + "\n")

    def close(self):
        self.f.close()


class ConsoleSink:
    """One status line per health tick, plus a line whenever a check changes."""

    def __init__(self):
        self.last_fused = None

    def emit(self, kind, rec):
        if kind == "fused":
            self.last_fused = rec
            return
        if kind != "health":
            return
        for name, st in rec["changed"].items():
            c = rec["checks"][name]
            if st != OK or len(rec["changed"]) < len(rec["checks"]):
                print(f"  [{st:4}] {name}: {c['value']} ({c['detail']})", flush=True)
        f = self.last_fused or {}
        c = rec["checks"]
        g = lambda k: c.get(k, {}).get("value")  # noqa: E731
        hot = (f"hottest {f['max_value']:.0f} at {f['max_bearing_deg']:+.1f} deg, {f['max_range_m']:.2f} m"
               if f.get("max_value") is not None else "no returns in image")
        print(f"[{rec['overall']:4}] lidar {g('lidar_rate')} pts/s @ {g('spin_rate')} Hz | "
              f"thermal {g('thermal_fps')} fps | latency p95 {g('fusion_latency')} ms | {hot}", flush=True)


class StatusServer:
    """GET /status -> latest health + latest fused summary as JSON."""

    def __init__(self, port: int, engine: "LiveEngine"):
        eng = engine

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/") not in ("", "/status"):
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({"health": eng.health.latest, "fused": eng.last_fused}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------


class LiveEngine:
    def __init__(self, cfg: dict, sinks: list, replay: Optional[Path] = None, speed: float = 1.0,
                 faults: Optional[Faults] = None, status_port: Optional[int] = None):
        self.cfg, self.sinks = cfg, sinks
        self.buf = LidarBuffer(keep_s=3.0)
        self.frames: queue.Queue = queue.Queue(maxsize=4)
        self.last_fused = None
        self.speed = speed if replay else 1.0
        if replay:
            replay = Path(replay)
            lid = pd.read_csv(replay / "lidar_points.csv", usecols=["t_ns"])
            fr = pd.read_csv(replay / "thermal" / "frames.csv", usecols=["t_ns"])
            clock = ReplayClock(int(min(lid.t_ns.min(), fr.t_ns.min())), self.speed)
            f = faults or Faults()
            self.lidar = LidarReplaySource(replay / "lidar_points.csv", self.buf, clock, f)
            self.camera = ThermalReplaySource(replay / "thermal", self.frames, clock, f)
        else:
            lc, tc = cfg["lidar"], cfg["thermal"]
            self.lidar = LidarSerialSource(lc["port"], self.buf, lc.get("baud", 230400))
            self.camera = CameraSource(tc["device"], self.frames, tc.get("raw16", False),
                                       tc.get("width"), tc.get("height"), tc.get("fps"))
        self.fuser = Fuser(cfg, self.buf, self.frames, self._emit, self.speed)
        self.health = HealthMonitor(cfg, self.buf, self.lidar, self.camera, self.fuser, self._emit, self.speed)
        self.server = StatusServer(status_port, self) if status_port else None

    def _emit(self, kind, rec):
        if kind == "fused":
            self.last_fused = rec
        for s in self.sinks:
            try:
                s.emit(kind, rec)
            except Exception as e:  # a broken sink must not stop the pipeline
                print(f"sink error ({type(s).__name__}): {e}", flush=True)

    def start(self):
        for t in (self.lidar, self.camera, self.fuser, self.health):
            t.start()
        if self.server:
            self.server.start()

    def stop(self):
        for t in (self.lidar, self.camera, self.fuser, self.health):
            t.stop()
        for t in (self.lidar, self.camera, self.fuser, self.health):
            t.join(timeout=3)
        if self.server:
            self.server.stop()

    def replay_done(self) -> bool:
        return not self.lidar.is_alive() and not self.camera.is_alive() and self.frames.empty()
