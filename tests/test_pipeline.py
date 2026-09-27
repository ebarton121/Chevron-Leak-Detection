"""Run: python -m pytest tests -q   (or python tests/test_pipeline.py)"""
import io
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leak_sync import lidar_ld  # noqa: E402
from leak_sync.repair_export import repair  # noqa: E402
from leak_sync.sync import associate, estimate_time_offset  # noqa: E402


def make_packet(speed, start_cdeg, end_cdeg, ts_ms, dists, ints):
    body = struct.pack("<BBHH", 0x54, 0x2C, speed, start_cdeg)
    for d, i in zip(dists, ints):
        body += struct.pack("<HB", d, i)
    body += struct.pack("<HH", end_cdeg, ts_ms)
    return body + bytes([lidar_ld.crc8(body)])


def test_parser_wrap_and_resync():
    # packet crossing 359 -> 0, preceded by junk and followed by a corrupt packet
    good = make_packet(3600, 35500, 400, 29990, [1000] * 12, [200] * 12)
    bad = bytearray(good); bad[10] ^= 0xFF
    stream = io.BytesIO(b"\x00\x13\x54" + good + bytes(bad) + good)
    reads = iter(lambda: stream.read(512) or None, None)
    pkts = list(lidar_ld.iter_packets(lambda n: next(reads, None)))
    assert len(pkts) == 2, len(pkts)                      # corrupt one dropped
    a = pkts[0].angles()
    assert abs(a[0] - 355) < 1e-6 and abs(a[-1] - 4) < 1e-6  # 9 deg span, not 351
    assert all(abs(((a[i + 1] - a[i]) % 360) - 9 / 11) < 1e-6 for i in range(11))


def test_repair_real_export():
    p = Path("/mnt/user-data/uploads/lidar_output_9_21_A1.xlsx")
    if not p.exists():
        return
    out, rep = repair(pd.read_excel(p))
    step = out.groupby("packet").angle_deg.apply(lambda a: (np.diff(a) % 360).max()).dropna()
    assert step.max() < 1.5, step.max()                    # was 15-45 deg
    assert (np.diff(out.t_ns) >= 0).all()
    print(rep)


def test_association_and_offset_recovery():
    # 10 Hz lidar, 9 Hz camera whose timestamps lag reality by 120 ms.
    rng = np.random.default_rng(0)
    t_l = np.arange(0, 20, 1 / 4500)
    ang = (t_l * 3600) % 360
    rev = (t_l * 10).astype(int)
    present = (np.sin(2 * np.pi * t_l / 3.1) > 0.3)          # warm object in/out
    dist = np.where(((ang > 355) | (ang < 5)) & present, 400, 2000) + rng.normal(0, 5, len(t_l))
    lidar = pd.DataFrame({"t_ns": (t_l * 1e9).astype(np.int64), "rev": rev,
                          "angle_deg": ang, "distance_mm": dist})
    t_f = np.arange(0, 20, 1 / 9)
    true_present = np.sin(2 * np.pi * t_f / 3.1) > 0.3
    frames = pd.DataFrame({"frame_idx": np.arange(len(t_f)),
                           "t_ns": ((t_f + 0.120) * 1e9).astype(np.int64),
                           "value": true_present * 50 + rng.normal(0, 2, len(t_f))})
    est = estimate_time_offset(lidar, frames, (350, 10))
    assert abs(est["thermal_time_offset_s"] - (-0.120)) <= 0.02, est
    a = associate(lidar, frames, 0.05, est["thermal_time_offset_s"])
    assert a.dt_ms.abs().max() <= 50
    print(est)


def test_sdcard_import_and_coarse_offset():
    """Camera clock 7.3 s off, TIFFs at 2 fps: import, then coarse+fine search."""
    import json, tempfile
    from PIL import Image
    from leak_sync.thermal_sdcard import import_tiffs
    rng = np.random.default_rng(1)
    tmp = Path(tempfile.mkdtemp()); sd = tmp / "sd"; sess = tmp / "sess"; sd.mkdir(); sess.mkdir()
    anchor = {"monotonic_ns": 5_000_000_000, "utc_ns": 1_790_000_000_000_000_000}
    (sess / "session_anchor.json").write_text(json.dumps(anchor))
    # irregular warm-object presence (not periodic, so no aliasing)
    edges = np.cumsum(rng.uniform(0.6, 2.5, 40)); edges = edges[edges < 60]
    present_at = lambda t: (np.searchsorted(edges, t) % 2 == 1)
    clock_err = 7.3
    for i, t in enumerate(np.arange(0, 60, 0.5)):   # true time since anchor
        val = 9000 + 800 * present_at(t) + rng.normal(0, 20)
        img = Image.fromarray(np.full((16, 16), val, dtype=np.uint16))
        exif = Image.Exif()
        utc = anchor["utc_ns"] / 1e9 + t + clock_err
        from datetime import datetime, timezone
        d = datetime.fromtimestamp(utc, tz=timezone.utc)
        exif[306] = d.strftime("%Y:%m:%d %H:%M:%S"); exif[37520] = f"{d.microsecond // 1000:03d}"
        img.save(sd / f"FLIR{i:04d}.tif", exif=exif)
    rep = import_tiffs(sd, sess)
    assert rep["frames"] == 120 and rep["time_source"] == ["exif"], rep
    frames = pd.read_csv(sess / "thermal" / "frames.csv")
    frames["value"] = [float(np.asarray(Image.open(sess / "thermal" / f)).mean()) for f in frames.file]
    t_l = np.arange(0, 60, 1 / 4500); ang = (t_l * 3600) % 360
    dist = np.where(((ang > 355) | (ang < 5)) & present_at(t_l), 400, 2000)
    lidar = pd.DataFrame({"t_ns": (anchor["monotonic_ns"] + t_l * 1e9).astype(np.int64),
                          "rev": (t_l * 10).astype(int), "angle_deg": ang, "distance_mm": dist})
    c = estimate_time_offset(lidar, frames, (350, 10), search_s=30, step_s=0.05)
    f = estimate_time_offset(lidar, frames, (350, 10), search_s=0.5, step_s=0.005,
                             center_s=c["thermal_time_offset_s"])
    assert abs(f["thermal_time_offset_s"] + clock_err) < 0.15, (c, f)
    print(rep, f)


if __name__ == "__main__":
    test_parser_wrap_and_resync(); test_repair_real_export(); test_association_and_offset_recovery()
    test_sdcard_import_and_coarse_offset()
    print("all tests passed")
