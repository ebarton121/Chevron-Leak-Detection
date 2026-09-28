"""Reader for LDRobot-protocol 2D LiDARs (LD06 / LD19 / STL-06P / STL-19P).

Packet (47 bytes, little-endian):
  0x54 | 0x2C | speed u16 deg/s | start_angle u16 (0.01 deg)
  12 x (distance u16 mm, intensity u8)
  end_angle u16 (0.01 deg) | timestamp u16 ms (wraps at 30000) | crc8

Compared with the lidar_output_9_21_A1 export, this reader (a) interpolates the
12 point angles correctly across the 359->0 wrap, (b) keeps the device timestamp
and (c) stamps every point on the shared host clock.
"""
from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .clock import DeviceClockMapper, now_ns

HEADER, VERLEN, PKT_LEN, N_PTS = 0x54, 0x2C, 47, 12
BAUD = 230400


def _crc_table(poly: int = 0x4D) -> List[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        table.append(c)
    return table


_CRC = _crc_table()


def crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c = _CRC[c ^ b]
    return c


@dataclass
class LidarPacket:
    speed_dps: int
    start_deg: float
    end_deg: float
    dev_ts_ms: int
    distances_mm: List[int]
    intensities: List[int]
    t_host_ns: int

    @property
    def step_deg(self) -> float:
        return ((self.end_deg - self.start_deg) % 360.0) / (N_PTS - 1)

    def angles(self) -> List[float]:
        return [(self.start_deg + i * self.step_deg) % 360.0 for i in range(N_PTS)]


def parse_packet(buf: bytes, t_host_ns: int) -> Optional[LidarPacket]:
    if len(buf) != PKT_LEN or buf[0] != HEADER or buf[1] != VERLEN:
        return None
    if crc8(buf[:-1]) != buf[-1]:
        return None
    speed, start = struct.unpack_from("<HH", buf, 2)
    dists, ints = [], []
    for i in range(N_PTS):
        d, it = struct.unpack_from("<HB", buf, 6 + 3 * i)
        dists.append(d)
        ints.append(it)
    end, ts = struct.unpack_from("<HH", buf, 42)
    return LidarPacket(speed, start / 100.0, end / 100.0, ts, dists, ints, t_host_ns)


def iter_packets(read: Callable[[int], Optional[bytes]], stats: Optional[dict] = None) -> Iterator[LidarPacket]:
    """`read(n)` returns bytes, b"" on timeout (keep going) or None to stop.
    Resyncs on the 0x54 0x2C header and drops CRC failures. If `stats` is
    given, stats["packets"] and stats["bad_packets"] are kept up to date."""
    if stats is not None:
        stats.setdefault("packets", 0)
        stats.setdefault("bad_packets", 0)
    buf = bytearray()
    while True:
        chunk = read(512)
        t = now_ns()
        if chunk is None:
            return
        buf.extend(chunk)
        while len(buf) >= PKT_LEN:
            i = buf.find(b"\x54\x2c")
            if i < 0:
                del buf[:-1]
                break
            if i:
                del buf[:i]
            if len(buf) < PKT_LEN:
                break
            pkt = parse_packet(bytes(buf[:PKT_LEN]), t)
            if pkt is None:
                if stats is not None:
                    stats["bad_packets"] += 1
                del buf[:1]
                continue
            del buf[:PKT_LEN]
            if stats is not None:
                stats["packets"] += 1
            yield pkt


POINT_FIELDS = ["t_ns", "rev", "angle_deg", "distance_mm", "intensity", "speed_dps"]


def packets_to_points(packets: Iterator[LidarPacket], mapper: Optional[DeviceClockMapper] = None) -> Iterator[dict]:
    """Device clock -> host clock per packet, then each point offset by its
    angular position at the reported rotation speed. `rev` increments each
    full revolution so a scan can be pulled out as one unit."""
    mapper = mapper or DeviceClockMapper(wrap=30000, units_per_s=1000.0)
    rev, last_start = 0, None
    for p in packets:
        t_last = mapper.map(p.dev_ts_ms, p.t_host_ns)  # timestamp ~ last point
        if last_start is not None and p.start_deg < last_start - 180:
            rev += 1
        last_start = p.start_deg
        dt_ns = p.step_deg / max(p.speed_dps, 1) * 1e9
        for i, ang in enumerate(p.angles()):
            yield {
                "t_ns": int(t_last - (N_PTS - 1 - i) * dt_ns),
                "rev": rev,
                "angle_deg": round(ang, 3),
                "distance_mm": p.distances_mm[i],
                "intensity": p.intensities[i],
                "speed_dps": p.speed_dps,
            }


def record(port: str, out_csv: Path, stop_event=None, baud: int = BAUD) -> int:
    import serial  # pip install pyserial

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with serial.Serial(port, baud, timeout=0.1) as ser, open(out_csv, "w", newline="") as f:
        def read(k):
            return None if (stop_event is not None and stop_event.is_set()) else ser.read(k)

        w = csv.DictWriter(f, fieldnames=POINT_FIELDS)
        w.writeheader()
        for pt in packets_to_points(iter_packets(read)):
            w.writerow(pt)
            n += 1
    return n
