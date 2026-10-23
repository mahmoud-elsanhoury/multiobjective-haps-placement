# -*- coding: utf-8 -*-
"""Reference HAPS geometry helpers used by the published experiment.

Only the two functions required by the final camera-ready computation are exposed
here. Their executable bodies are preserved from the validated source used in the
experiment; comments and documentation are adapted for public release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymap3d as pm

from .receiver_eval import _enu_dirs, occlusion_mask


def haps_ring_k(epoch_utc: pd.Timestamp, haps_cfg: dict,
                center_latlon: tuple[float, float], k: int) -> np.ndarray:
    """Nested 6-slot ring: slot i at azimuth phase0 + i*60 deg, i = 0..k-1.

    Radius/altitude/phase0 are identical to baseline (fixed drift radius, 20 km,
    phase0 = 2pi*sod/period). Returns a (k,3) array of (lon, lat, alt_m)
    geodetic — the same convention as baseline's haps_positions_k. For k=6 this is
    exactly baseline's k=6 ring.
    """
    lat0, lon0 = center_latlon
    drift = haps_cfg["quasi_stationary_drift"]
    period_s = float(drift["period_h"]) * 3600.0
    sod = float(epoch_utc.hour * 3600 + epoch_utc.minute * 60 + epoch_utc.second)
    phase0 = 2.0 * np.pi * (sod / period_s)
    r = float(drift["radius_m"])
    alt = float(haps_cfg["altitude_km"]) * 1000.0
    pos = []
    for i in range(k):
        phase = phase0 + i * (2.0 * np.pi / 6.0)   # i*60 deg (fixed slot index)
        e, n = r * np.cos(phase), r * np.sin(phase)
        lat, lon, _ = pm.enu2geodetic(e, n, 0.0, lat0, lon0, 0.0)
        pos.append([float(lon), float(lat), alt])
    return np.asarray(pos)


def _haps_los(prep: dict, scene, cfg: dict,
              haps_lonlatalt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HAPS az/el/range + LOS (occlusion-tested against the same scene)."""
    recs = prep["recs"]
    h_agl = float(cfg["grid"]["receiver_height_agl_m"])
    lat0 = recs["lat"].values
    lon0 = recs["lon"].values
    alt0 = np.full(len(recs), h_agl)
    n_h = haps_lonlatalt.shape[0]
    haz, hel, hrng = [], [], []
    for i in range(n_h):
        lon_h, lat_h, alt_h = haps_lonlatalt[i]
        x, y, z = pm.geodetic2ecef(lat_h, lon_h, alt_h)
        az, el, rng = pm.ecef2aer(x, y, z, lat0, lon0, alt0)
        haz.append(np.asarray(az, dtype=np.float64))
        hel.append(np.asarray(el, dtype=np.float64))
        hrng.append(np.asarray(rng, dtype=np.float64))
    haz = np.column_stack(haz)
    hel = np.column_stack(hel)
    hrng = np.column_stack(hrng)
    habove = hel >= prep["mask_deg"]
    rec_i, sv_i = np.where(habove)
    hblocked = occlusion_mask(scene, prep["origins"][rec_i],
                              _enu_dirs(haz[rec_i, sv_i], hel[rec_i, sv_i]),
                              hrng[rec_i, sv_i], prep["max_batch"])
    hlos = np.zeros_like(habove)
    hlos[rec_i[~hblocked], sv_i[~hblocked]] = True
    return haz, hel, hrng, hlos
