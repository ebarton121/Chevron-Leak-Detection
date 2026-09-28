"""Scene analysis helpers for LiDAR recordings."""
import numpy as np
import pandas as pd


def flag_stray_points(df: pd.DataFrame, closer_frac=0.85, min_mm=60) -> pd.Series:
    """True for 'stray' returns: closer than the usual background in that
    direction AND isolated (1-2 points, not part of a solid object).
    Spray or mist shows up as many of these, scattered and short-lived; a few
    that repeat in the same place every revolution are small static objects."""
    d = df[df.distance_mm > min_mm]
    b = d.angle_deg.astype(int) % 360
    base = d.groupby(b).distance_mm.quantile(0.75)
    fg = d[d.distance_mm < closer_frac * b.map(base)].sort_values(["rev", "angle_deg"])
    new = (fg.rev.diff() != 0) | (fg.angle_deg.diff().abs() > 2) | (fg.distance_mm.diff().abs() > 150)
    cl = new.cumsum()
    size = cl.map(cl.value_counts())
    out = pd.Series(False, index=df.index)
    out.loc[fg.index[size.to_numpy() <= 2]] = True
    return out


def summary(df: pd.DataFrame, stray: pd.Series) -> dict:
    d = df[df.distance_mm > 60]
    dur = (d.t_ns.max() - d.t_ns.min()) / 1e9
    nrev = d.rev.nunique()
    s = {"duration_s": round(float(dur), 1), "revolutions": int(nrev),
         "rotation_hz": round(float(d.speed_dps.mean()) / 360, 2) if "speed_dps" in d else None,
         "median_range_m": round(float(d.distance_mm.median()) / 1000, 2),
         "max_range_m": round(float(d.distance_mm.quantile(0.995)) / 1000, 1),
         "stray_per_rev": round(float(stray.sum()) / max(nrev, 1), 1)}
    if "packet" in df:
        ps = df.groupby("packet").size()
        lost = int((12 - ps[ps < 12]).sum())
        s["capture_loss_pct"] = round(100.0 * lost / (len(df) + lost), 1)
    near = df[(df.distance_mm > 0) & (df.distance_mm <= 60)]
    if len(near) > 0.01 * len(df):
        a = near.angle_deg
        s["blocked_deg"] = f"{int(a.quantile(0.05))}-{int(a.quantile(0.95))}"
    return s
