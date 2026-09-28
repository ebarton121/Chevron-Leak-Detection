# Chevron Leak Detection: LiDAR + Thermal Capture & Sync

Reads a 2D LiDAR (LDRobot LD06/LD19-protocol) and a USB thermal camera, stamps
both on one host clock, and produces a fused table: for each thermal frame, the
LiDAR returns captured within ±50 ms, projected into the image with the thermal
value sampled at each return.

## Layout
```
leak_sync/clock.py          shared monotonic clock + device-clock mapping
leak_sync/lidar_ld.py       serial packet parser (CRC, wrap-safe angles, timestamps)
leak_sync/thermal.py        live capture (USB capture dongle / webcam-class devices)
leak_sync/thermal_sdcard.py import Vue Pro R SD-card TIFF sequences
leak_sync/sync.py           association, projection, time-offset estimation
leak_sync/repair_export.py  fixes old lidar_output_*.xlsx exports
scripts/record.py           record both sensors together
scripts/fuse.py             build fused.csv for a session
scripts/visualize.py        interactive HTML viewer (2D + 3D) for a session or LiDAR CSV
scripts/live.py             REAL-TIME pipeline: live fusion + sensor health (or replay a session)
leak_sync/live.py           streaming engine: sources, fuser, health monitor, sinks
scripts/compare.py          side-by-side comparison of recordings (stats, scans, 3D, stray points)
leak_sync/analysis.py       stray-point (spray/mist candidate) detection, recording summaries
scripts/estimate_offset.py  measure thermal latency / camera clock error
scripts/import_sdcard.py    attach SD-card TIFFs to a LiDAR session
scripts/repair_export.py    CLI for the export repair
config/sensors.yaml         ports, time offset, intrinsics, extrinsics
tests/test_pipeline.py      parser, repair, sync and offset-recovery tests
```

## Thermal camera: FLIR Vue Pro R 640 (9 mm, 30 Hz)
The Vue Pro R does not stream over USB, so there are two capture paths:

**A. Live (default, best sync).** Camera HDMI or analog out -> USB capture
dongle -> laptop. `scripts/record.py` records it alongside the LiDAR on one
clock. 8-bit video, 640x480, 30 Hz; brightness, not temperature.

**B. SD card (real 14-bit values, ~2 fps).** Record a TIFF sequence on the
camera while `record.py` records the LiDAR, then:
```bash
python scripts/import_sdcard.py E:/DCIM/100_0001 data/session_01
python scripts/estimate_offset.py data/session_01 --roi X Y W H --search 30
```
Set the camera clock to UTC beforehand; `--search` absorbs the rest of the
clock error. For the sync calibration, move the warm object in and out at
irregular intervals (a regular rhythm can line up at the wrong offset).

## Quick start
```bash
pip install -r requirements.txt
python tests/test_pipeline.py
# edit config/sensors.yaml (port, camera index)
python scripts/record.py --out data/session_01 --seconds 30
python scripts/fuse.py data/session_01
python scripts/visualize.py data/session_01   # opens viewer.html (3D panel needs internet)
```

## Real-time mode
```bash
python scripts/live.py --out data/live_01 --status-port 8765          # live sensors
python scripts/live.py --replay data/session_01 --speed 2               # replay, no hardware
python scripts/live.py --replay data/session_01 --faults "lidar_gap:3-4,frozen:6-8"
```
Each thermal frame is fused as soon as the LiDAR points up to 50 ms after it
have arrived (typical frame-in to result-out: ~60 ms). Every second the health
monitor checks LiDAR data/rate/gaps/spin rate/serial corruption/clock drift,
thermal data/frame rate/gaps/frozen video, dropped frames, fusion latency and
LiDAR-camera association, and reports OK / WARN / FAIL. Results go to the
screen, `live.jsonl`, and `http://localhost:<port>/status`. Thresholds are in
the `health:` section of config/sensors.yaml. Other outputs (e.g. Kafka) plug
in as sinks: any object with `emit(kind, record)`.

## Sync approach
1. **One clock.** Both threads stamp with `time.monotonic_ns()` on the same machine.
2. **LiDAR jitter removal.** The LD06 packet carries a ms counter; it's mapped to
   host time with a running-minimum offset, so USB delivery jitter drops out.
   Each of the 12 points in a packet then gets its own time from the rotation speed.
3. **Thermal latency.** Frames are stamped on arrival, which is later than
   exposure. `thermal_time_offset_s` corrects this (same role as FAST-LIVO2's
   `img_time_offset`). Measure it: record while moving a warm object in and out
   of the LiDAR plane, then run `scripts/estimate_offset.py`.
4. **Association.** Points within `max_dt_s` of each corrected frame time.

## Calibration still needed (values in sensors.yaml are placeholders)
* Thermal intrinsics: heated/cooled checkerboard, OpenCV calibrateCamera.
* LiDAR->camera extrinsic (Rcl, Pcl): FAST-LIVO2 key names are used so
  FAST-Calib output (or a hand-measured mount) can be pasted in.

## Relation to FAST-LIVO2
FAST-LIVO2 (ROS1) needs a 3D LiDAR, an IMU and a hardware-triggered camera;
it will not run on a single-line 2D LiDAR. This repo borrows its config
conventions so an upgrade path (e.g. Livox + IMU) keeps the same calibration
files.

## third_party/FAST-LIVO2
Unmodified copy of [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2),
licensed **GPL-2.0** (see `third_party/FAST-LIVO2-main/LICENSE`). It is kept
separate from `leak_sync/` and is not imported by it. Currently used as a
reference for calibration/config conventions; it requires a 3D LiDAR + IMU and
does not run with the current 2D LiDAR.
