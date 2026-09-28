"""Detect cold / warm changes in a sequence of radiometric thermal frames.

Each frame is compared with a baseline (median of the first `baseline_n`
frames). Uncooled thermal cameras and rooms drift, so the whole-frame median
change ("drift") is removed first; what remains are LOCAL changes.

  * cold change  (< -threshold): evaporative cooling from sprays, liquid
    leaks, expanding gas (Joule-Thomson) - the leak-type signal.
  Pixels hidden by something warm in the baseline frames are excluded from
  cold detection (see occluder_c).
  * warm change  (> +threshold): people, hands, warm objects, heat left on
    surfaces (footprints, chairs) - usually NOT leaks, reported separately.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def analyze(arr: np.ndarray, baseline_n: int = 5, threshold_c: float = 1.5, min_blob_px: int = 400,
            occluder_c: float = 3.0, min_area_pct: float = 1.0, persist: int = 3):
    # Pixels covered by something much warmer than the scene (a person, an arm
    # near the lens) don't show the background. Leave them out of the baseline
    # so that when the warm thing moves away the spot doesn't look "colder".
    occ = np.stack([ndimage.binary_dilation(f > np.median(f) + occluder_c, iterations=6) for f in arr])
    masked = np.where(occ, np.nan, arr)
    base = np.nanmedian(masked[:baseline_n], axis=0)
    fill = np.isnan(base)                       # hidden in every baseline frame:
    if fill.any():                              # use later frames where visible
        later = np.nanmedian(masked, axis=0)
        base[fill] = later[fill]
    no_base = np.isnan(base)
    base = np.where(no_base, np.median(arr[:baseline_n], axis=0), base)
    rows, cold_masks, warm_masks, deltas = [], [], [], []
    for i, f in enumerate(arr):
        d = f - base
        drift = float(np.median(d))
        dc = d - drift
        cold = _clean((dc < -threshold_c) & ~no_base, min_blob_px)
        warm = _clean(dc > threshold_c, min_blob_px)
        ys, xs = np.nonzero(cold)
        rows.append({
            "scan_index": i,
            "drift_c": round(drift, 2),
            "cold_area_pct": round(100 * float(cold.mean()), 2),
            "cold_mean_drop_c": round(float(dc[cold].mean()), 2) if cold.any() else 0.0,
            "cold_max_drop_c": round(float(dc[cold].min()), 2) if cold.any() else 0.0,
            "cold_centroid_px": [int(xs.mean()), int(ys.mean())] if len(xs) else None,
            "cold_bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "warm_area_pct": round(100 * float(warm.mean()), 2),
            "scene_mean_c": round(float(f.mean()), 2),
        })
        cold_masks.append(cold)
        warm_masks.append(warm)
        deltas.append(dc)
    rows[0]["no_baseline_pct"] = round(100 * float(no_base.mean()), 1)
    rows[0]["baseline_from_later_frames_pct"] = round(100 * float(fill.mean()), 1)
    # Onset: first scan of a run of >= persist scans with cold area >= min_area_pct
    onset, run = None, 0
    for r in rows:
        run = run + 1 if r["cold_area_pct"] >= min_area_pct else 0
        if run >= persist:
            onset = r["scan_index"] - persist + 1
            break
    return rows, np.stack(deltas), np.stack(cold_masks), np.stack(warm_masks), onset


def _clean(mask, min_px):
    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= min_px)[0])
    return keep
