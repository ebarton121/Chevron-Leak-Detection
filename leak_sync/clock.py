"""One clock for every sensor.

Every record written by this package carries a timestamp from time.monotonic_ns()
on the SAME machine. Monotonic (not wall-clock) so NTP adjustments can't break
sync. A wall-clock anchor is saved once per session to map back to UTC.
"""
import json
import time
from pathlib import Path


def now_ns() -> int:
    return time.monotonic_ns()


def write_session_anchor(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = {"monotonic_ns": time.monotonic_ns(), "utc_ns": time.time_ns()}
    (out_dir / "session_anchor.json").write_text(json.dumps(anchor, indent=2))
    return anchor


class DeviceClockMapper:
    """Map a wrapping device counter (LD06: ms, wraps at 30000) onto host time.

    Serial/USB delivery jitter only ever makes data arrive LATE, so the running
    minimum of (host - device) is the best offset estimate. This removes jitter
    and gives smooth, monotonic per-packet timestamps.
    """

    def __init__(self, wrap: int, units_per_s: float, window: int = 2000):
        self.wrap = wrap
        self.scale = 1e9 / units_per_s
        self.window = window
        self._last_raw = None
        self._wraps = 0
        self._offsets = []
        self.offset_ns = None  # current device->host offset, for drift monitoring

    def map(self, device_raw: int, host_ns: int) -> int:
        if self._last_raw is not None and device_raw < self._last_raw - self.wrap // 2:
            self._wraps += 1
        self._last_raw = device_raw
        dev_ns = int((device_raw + self._wraps * self.wrap) * self.scale)
        self._offsets.append(host_ns - dev_ns)
        if len(self._offsets) > self.window:
            self._offsets.pop(0)
        self.offset_ns = min(self._offsets)
        return dev_ns + self.offset_ns
