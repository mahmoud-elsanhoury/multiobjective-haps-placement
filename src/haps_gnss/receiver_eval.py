# -*- coding: utf-8 -*-
"""Steps 3-4 of the published experiment protocol: receiver skyline/LOS availability,
DOP / CRLB / PEB on the receiver grid, and the HAPS overlay delta.

Implementation notes (fixed parameters only, no invented values):
- Epochs: config `skyplot.time_window_utc` sampled every `time_step_s`.
- Elevation mask: config `dop.elevation_mask_deg` (= `skyplot.elevation_mask_deg`).
- Time scales: receiver epochs are UTC; broadcast ToC/Toe are GNSS system
  times. Offsets are read from the BRDC header itself (LEAP SECONDS, GPUT,
  GAGP TIME SYSTEM CORR rows); documented standard values are used only if a
  row is absent. GPS: t_GPST = t_UTC + leap + GPUT_A0. Galileo:
  t_GST = t_GPST + GAGP_A0 (RINEX 3.05 TIME SYSTEM CORR semantics).
- URE model: sigma_i(el) = sigma_0,i / sin(el); sigma_0,i = RINEX URA (GPS
  SVacc, metres) or Galileo SISA (metres) when finite and > 0, else the
  fixed default `crlb.sigma_ure_default_m` (config `sigma_ure_source`).
- DOP metrics use pure elevation weighting (config dop.weighting = elevation);
  CRLB/PEB use the sigma-weighted covariance per config `crlb`.
- HAPS overlay (step 4): quasi-stationary platform at config altitude over
  the ROI center with the fixed circular drift (radius/period). For k
  platforms (count_sweep) the phases are phase_0 + 2*pi*i/k at the epoch
  phase phase_0 = 2*pi * sod / period_s (deterministic placement using only
  fixed parameters). HAPS pseudorange sigma: dense_urban value (all fixed
  ROIs are dense-urban cores). HAPS rays are occlusion-tested like GNSS rays.
- No DEM is applied: receiver heights are AGL above z = 0 (recorded as an
  implementation limitation in the execution report).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymap3d as pm

from .dop import geometry_matrix, math_sqrt
from .ephemeris import SatState, sv_position_ecef

log = logging.getLogger("haps_pnt.receiver_eval")

# documented standard fallbacks (used only if the BRDC header row is absent)
_STD_LEAP_S = 18.0


# --------------------------------------------------------------------------- #
# Time scales                                                                 #
# --------------------------------------------------------------------------- #
def time_system_offsets(nav_path: Path) -> dict:
    """LEAP SECONDS + GPUT/GAGP A0 offsets from a RINEX-3 NAV header."""
    leap, gput_a0, gagp_a0 = None, 0.0, 0.0
    with open(nav_path, "r", encoding="utf-8") as fh:
        for line in fh:
            kind = line[60:].strip()
            if kind == "LEAP SECONDS":
                try:
                    leap = float(line[:6])
                except ValueError:
                    pass
            elif kind == "TIME SYSTEM CORR":
                code = line[:4]
                try:
                    a0 = float(line[5:22].replace("D", "E"))
                except ValueError:
                    continue
                if code == "GPUT":
                    gput_a0 = a0
                elif code == "GAGP":
                    gagp_a0 = a0
            elif kind == "END OF HEADER":
                break
    if leap is None:
        leap = _STD_LEAP_S
    return {"leap_s": leap, "gput_a0_s": gput_a0, "gagp_a0_s": gagp_a0}


def sod_system_time(epoch_utc: pd.Timestamp, system: str, offs: dict) -> float:
    """Seconds-of-week of the UTC epoch in the SV's system time scale."""
    sod = float(epoch_utc.hour * 3600 + epoch_utc.minute * 60 + epoch_utc.second)
    t_gps = sod + offs["leap_s"] + offs["gput_a0_s"]
    if system == "E":
        return t_gps + offs["gagp_a0_s"]
    return t_gps


# --------------------------------------------------------------------------- #
# SV geometry at one epoch                                                    #
# --------------------------------------------------------------------------- #
def sv_state_from_row(row) -> SatState:
    return SatState(
        svid=int(str(row["sv"])[1:]),
        toe=float(row["toe"]),
        sqrt_a=float(row["sqrt_a"]),
        ecc=float(row["ecc"]),
        i0=float(row["i0"]),
        omega0=float(row["omega0"]),
        arg_perigee=float(row["arg_perigee"]),
        m0=float(row["m0"]),
        delta_n=float(row["delta_n"]),
        cus=float(row["cus"]), cuc=float(row["cuc"]),
        crs=float(row["crs"]), crc=float(row["crc"]),
        cis=float(row["cis"]), cic=float(row["cic"]),
        idot=float(row["idot"]),
        omega_dot=float(row["omega_dot"]),
    )


def sv_sigma_ure(row, default_m: float) -> float:
    """Frozen sigma_ure_source: RINEX URA / Galileo SISA (m) else default."""
    v = row.get("sv_acc") if str(row["sv"]).startswith("G") else row.get("sisa")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default_m)
    if np.isfinite(v) and v > 0:
        return v
    return float(default_m)


def propagate_svs(sel: pd.DataFrame, epoch_utc: pd.Timestamp, offs: dict) -> dict:
    """sv -> ECEF position (m) at the epoch, using the selected ephemerides."""
    out = {}
    for _, row in sel.iterrows():
        sysc = str(row["sv"])[0]
        t_sow = sod_system_time(epoch_utc, sysc, offs)
        try:
            p = sv_position_ecef(sv_state_from_row(row), t_sow, t_sow)
        except Exception as exc:
            log.warning("propagation failed for %s: %s", row["sv"], exc)
            continue
        if not np.all(np.isfinite(p)):
            continue
        out[str(row["sv"])] = (p, float(row["sigma_ure_m"]))
    return out


# --------------------------------------------------------------------------- #
# Receiver grid                                                               #
# --------------------------------------------------------------------------- #
def receiver_grid(roi: dict, grid_cfg: dict, frame) -> pd.DataFrame:
    """Regular receiver grid over the fixed ROI bbox (local metres)."""
    lat_min, lon_min, lat_max, lon_max = roi["bbox"]
    spacing = float(grid_cfg["receiver_spacing_m"])
    x0, y0 = frame.to_local(lon_min, lat_min)
    x1, y1 = frame.to_local(lon_max, lat_max)
    xs = x0 + spacing * np.arange(int(np.floor((x1 - x0) / spacing)) + 1)
    ys = y0 + spacing * np.arange(int(np.floor((y1 - y0) / spacing)) + 1)
    xs = xs[xs <= x1 + 1e-9]
    ys = ys[ys <= y1 + 1e-9]
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = frame.to_wgs84(gx.ravel(), gy.ravel())
    return pd.DataFrame({"x_m": gx.ravel(), "y_m": gy.ravel(),
                         "lon": np.asarray(lon), "lat": np.asarray(lat)})


# --------------------------------------------------------------------------- #
# Ray tracing (embrex)                                                        #
# --------------------------------------------------------------------------- #
def build_scene(mesh):
    from embreex.rtcore_scene import EmbreeScene
    from embreex.mesh_construction import TriangleMesh
    scene = EmbreeScene()
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    TriangleMesh(scene, verts, faces)
    return scene


def occlusion_mask(scene, origins: np.ndarray, dirs: np.ndarray,
                   dists: np.ndarray, max_batch: int) -> np.ndarray:
    """True where the ray (origin -> origin + dist*dir) is blocked."""
    n = origins.shape[0]
    blocked = np.zeros(n, dtype=bool)
    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        prim = scene.run(np.ascontiguousarray(origins[i:j], dtype=np.float32),
                         np.ascontiguousarray(dirs[i:j], dtype=np.float32),
                         dists=np.ascontiguousarray(dists[i:j], dtype=np.float32),
                         query="OCCLUDED")
        blocked[i:j] = np.asarray(prim) >= 0
    return blocked


# --------------------------------------------------------------------------- #
# HAPS overlay                                                                #
# --------------------------------------------------------------------------- #
def haps_geodetic(epoch_utc: pd.Timestamp, haps_cfg: dict,
                  center_latlon: tuple[float, float]) -> tuple[float, float, float]:
    """Quasi-stationary HAPS position (geodetic) at the epoch (drift phase 0)."""
    lat0, lon0 = center_latlon
    period_s = float(haps_cfg["quasi_stationary_drift"]["period_h"]) * 3600.0
    sod = float(epoch_utc.hour * 3600 + epoch_utc.minute * 60 + epoch_utc.second)
    phase = 2.0 * np.pi * (sod / period_s)
    r = float(haps_cfg["quasi_stationary_drift"]["radius_m"])
    e, n = r * np.cos(phase), r * np.sin(phase)
    lat, lon, _ = pm.enu2geodetic(e, n, 0.0, lat0, lon0, 0.0)
    return float(lat), float(lon), float(haps_cfg["altitude_km"]) * 1000.0


def haps_positions_k(epoch_utc: pd.Timestamp, haps_cfg: dict,
                     center_latlon: tuple[float, float], k: int) -> np.ndarray:
    """k HAPS platforms: deterministic equally spaced phases at the epoch."""
    lat0, lon0 = center_latlon
    period_s = float(haps_cfg["quasi_stationary_drift"]["period_h"]) * 3600.0
    sod = float(epoch_utc.hour * 3600 + epoch_utc.minute * 60 + epoch_utc.second)
    phase0 = 2.0 * np.pi * (sod / period_s)
    r = float(haps_cfg["quasi_stationary_drift"]["radius_m"])
    alt = float(haps_cfg["altitude_km"]) * 1000.0
    pos = []
    for i in range(k):
        phase = phase0 + 2.0 * np.pi * i / k
        e, n = r * np.cos(phase), r * np.sin(phase)
        lat, lon, _ = pm.enu2geodetic(e, n, 0.0, lat0, lon0, 0.0)
        pos.append([float(lon), float(lat), alt])   # (lon, lat, alt) geodetic
    return np.asarray(pos)


# --------------------------------------------------------------------------- #
# Per-epoch evaluation                                                        #
# --------------------------------------------------------------------------- #
def _enu_dirs(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.column_stack([np.sin(az) * np.cos(el),
                            np.cos(az) * np.cos(el),
                            np.sin(el)])


def prepare_epoch(epoch_utc: pd.Timestamp, sv_geom: dict, recs: pd.DataFrame,
                  scene, cfg: dict) -> dict:
    """GNSS geometry + occlusion at one epoch (reused by base and HAPS runs)."""
    mask_deg = float(cfg["dop"]["elevation_mask_deg"])
    h_agl = float(cfg["grid"]["receiver_height_agl_m"])
    max_batch = int(cfg["raytracing"]["max_rays_per_batch"])
    lat0 = recs["lat"].values
    lon0 = recs["lon"].values
    alt0 = np.full(len(recs), h_agl)
    origins = np.column_stack([recs["x_m"].values, recs["y_m"].values,
                               np.full(len(recs), h_agl)]).astype(np.float64)

    sv_names, sv_az, sv_el, sv_rng, sv_sig = [], [], [], [], []
    for sv, (p, sig) in sv_geom.items():
        az, el, rng = pm.ecef2aer(p[0], p[1], p[2], lat0, lon0, alt0)
        sv_names.append(sv)
        sv_az.append(np.asarray(az, dtype=np.float64))
        sv_el.append(np.asarray(el, dtype=np.float64))
        sv_rng.append(np.asarray(rng, dtype=np.float64))
        sv_sig.append(float(sig))

    n_rec = len(recs)
    prep = {"epoch": epoch_utc, "recs": recs, "origins": origins,
            "sv_names": sv_names, "n_rec": n_rec, "mask_deg": mask_deg,
            "max_batch": max_batch}
    if not sv_names:
        prep.update({"azm": np.zeros((n_rec, 0)), "elm": np.zeros((n_rec, 0)),
                     "rngm": np.zeros((n_rec, 0)), "sigv": np.zeros(0),
                     "los": np.zeros((n_rec, 0), dtype=bool),
                     "above": np.zeros((n_rec, 0), dtype=bool)})
        return prep
    azm = np.column_stack(sv_az)
    elm = np.column_stack(sv_el)
    rngm = np.column_stack(sv_rng)
    above = elm >= mask_deg
    rec_i, sv_i = np.where(above)
    blocked = occlusion_mask(scene, origins[rec_i],
                             _enu_dirs(azm[rec_i, sv_i], elm[rec_i, sv_i]),
                             rngm[rec_i, sv_i], max_batch)
    los = np.zeros_like(above)
    los[rec_i[~blocked], sv_i[~blocked]] = True
    prep.update({"azm": azm, "elm": elm, "rngm": rngm,
                 "sigv": np.asarray(sv_sig, dtype=np.float64),
                 "above": above, "los": los})
    return prep


def _metrics_batched(ok: np.ndarray, los: np.ndarray, azm: np.ndarray,
                     elm: np.ndarray, sigv: np.ndarray, k_chi2: float) -> dict | None:
    """Fully vectorized (n_rec,4,4) batched inverse, mathematically identical
    to the sequential per-receiver path. LOS acts as a 0/1 weight multiplier.
    Receivers outside `ok` get identity matrices (results are discarded).
    Returns None to request the strict sequential fallback (any singular or
    non-finite batch result)."""
    n_rec, n_sat = azm.shape
    az = np.radians(azm)
    el = np.radians(elm)
    cosel = np.cos(el)
    G = np.empty((n_rec, n_sat, 4), dtype=np.float64)
    G[..., 0] = np.sin(az) * cosel
    G[..., 1] = np.cos(az) * cosel
    G[..., 2] = np.sin(el)
    G[..., 3] = 1.0
    s2 = np.sin(el) ** 2
    w_el = np.where(los, s2, 0.0)                    # DOP weights (elevation)
    w_sig = np.where(los, s2 / sigv[None, :] ** 2, 0.0)  # CRLB weights
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            M1 = np.einsum("nki,nk,nkj->nij", G, w_el, G)
            M2 = np.einsum("nki,nk,nkj->nij", G, w_sig, G)
            excl = np.ones(n_rec, dtype=bool)
            excl[ok] = False
            if excl.any():
                eye = np.eye(4, dtype=np.float64)
                M1[excl] = eye
                M2[excl] = eye
            cw = np.linalg.inv(M1)
            c = np.linalg.inv(M2)
            hdop = np.sqrt(cw[:, 0, 0] + cw[:, 1, 1])
            vdop = np.sqrt(cw[:, 2, 2])
            pdop = np.sqrt(cw[:, 0, 0] + cw[:, 1, 1] + cw[:, 2, 2])
            gdop = np.sqrt(np.einsum("nii->n", cw))
            tdop = np.sqrt(cw[:, 3, 3])
            a, cc, b = c[:, 0, 0], c[:, 1, 1], c[:, 0, 1]
            lam = 0.5 * (a + cc) + np.sqrt(0.25 * (a - cc) ** 2 + b * b)
            peb = np.sqrt(k_chi2 * np.maximum(lam, 0.0))
    except np.linalg.LinAlgError:
        return None
    vals = np.vstack([hdop[ok], vdop[ok], pdop[ok], gdop[ok], tdop[ok], peb[ok]])
    if np.isinf(vals).any():
        return None
    return {"hdop": hdop[ok], "vdop": vdop[ok], "pdop": pdop[ok],
            "gdop": gdop[ok], "tdop": tdop[ok], "peb": peb[ok]}


def _metrics_arrays(prep: dict, los: np.ndarray, azm: np.ndarray, elm: np.ndarray,
                    sigv: np.ndarray, confidence: float) -> dict:
    """Per-receiver DOP / CRLB / PEB from a LOS matrix (vectorized)."""
    n_rec = prep["n_rec"]
    n_above = prep["above"].sum(axis=1) if prep["above"].shape[1] else np.zeros(n_rec)
    n_los = los.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        availability = np.where(n_above > 0, n_los / np.maximum(n_above, 1), np.nan)
    hdop = np.full(n_rec, np.nan); vdop = np.full(n_rec, np.nan)
    pdop = np.full(n_rec, np.nan); gdop = np.full(n_rec, np.nan)
    tdop = np.full(n_rec, np.nan); peb = np.full(n_rec, np.nan)
    k = float(chi2_ppf(confidence))
    ok = np.where(n_los >= 4)[0]
    res = _metrics_batched(ok, los, azm, elm, sigv, k) if ok.size else None
    if ok.size and res is not None:
        hdop[ok] = res["hdop"]; vdop[ok] = res["vdop"]; pdop[ok] = res["pdop"]
        gdop[ok] = res["gdop"]; tdop[ok] = res["tdop"]; peb[ok] = res["peb"]
    else:
        for r in ok:
            idx = np.where(los[r])[0]
            el_r = elm[r, idx]
            az_r = azm[r, idx]
            g = geometry_matrix(el_r, az_r)
            sin2 = np.sin(np.radians(el_r)) ** 2
            try:
                # elevation-weighted DOP (dimensionless)
                cw = np.linalg.inv((g * np.sqrt(sin2)[:, None]).T @ (g * np.sqrt(sin2)[:, None]))
                hdop[r] = math_sqrt(cw[0, 0] + cw[1, 1])
                vdop[r] = math_sqrt(cw[2, 2])
                pdop[r] = math_sqrt(cw[0, 0] + cw[1, 1] + cw[2, 2])
                gdop[r] = math_sqrt(np.trace(cw))
                tdop[r] = math_sqrt(cw[3, 3])
                # sigma-weighted CRLB covariance (m^2)
                w = sin2 / sigv[idx] ** 2
                c = np.linalg.inv((g * np.sqrt(w)[:, None]).T @ (g * np.sqrt(w)[:, None]))
                a, cc, b = c[0, 0], c[1, 1], c[0, 1]
                lam = 0.5 * (a + cc) + math.sqrt(0.25 * (a - cc) ** 2 + b * b)
                peb[r] = math.sqrt(k * max(lam, 0.0))
            except np.linalg.LinAlgError:
                continue
    return {"availability": availability, "n_above": n_above, "n_los": n_los,
            "hdop": hdop, "vdop": vdop, "pdop": pdop, "gdop": gdop,
            "tdop": tdop, "peb95_m": peb}


def chi2_ppf(confidence: float) -> float:
    from scipy.stats import chi2
    return float(chi2.ppf(confidence, df=2))


def _assemble_frame(prep: dict, m: dict) -> pd.DataFrame:
    recs = prep["recs"]
    return pd.DataFrame({
        "epoch": str(prep["epoch"]),
        "rec_id": np.arange(prep["n_rec"]),
        "lon": recs["lon"].values, "lat": recs["lat"].values,
        "n_above_mask": m["n_above"].astype(int),
        "n_los": m["n_los"].astype(int),
        "availability": m["availability"],
        "hdop": m["hdop"], "vdop": m["vdop"], "pdop": m["pdop"],
        "gdop": m["gdop"], "tdop": m["tdop"], "peb95_m": m["peb95_m"],
    })


def evaluate_prepared(prep: dict, cfg: dict,
                      haps_xyz_lonlatalt: np.ndarray | None = None,
                      scene=None) -> pd.DataFrame:
    """Metrics from prepared GNSS geometry; reuses the GNSS occlusion/LOS work
    when adding HAPS overlays (scene required only for HAPS occlusion)."""
    crlb_cfg = cfg["crlb"]
    confidence = float(crlb_cfg["peb_confidence"])
    if haps_xyz_lonlatalt is None:
        return _assemble_frame(prep, _metrics_arrays(
            prep, prep["los"], prep["azm"], prep["elm"], prep["sigv"], confidence))
    return _evaluate_haps_augmented(prep, cfg, scene, haps_xyz_lonlatalt, confidence)


def _evaluate_haps_augmented(prep: dict, cfg: dict, scene,
                             haps_xyz_lonlatalt: np.ndarray,
                             confidence: float) -> pd.DataFrame:
    """Augmented metrics: GNSS LOS set (prepared) + k HAPS rays."""
    recs = prep["recs"]
    h_agl = float(cfg["grid"]["receiver_height_agl_m"])
    haps_sig = float(cfg["haps_overlay"]["pseudorange_sigma_m"]["dense_urban"])
    lat0 = recs["lat"].values
    lon0 = recs["lon"].values
    alt0 = np.full(len(recs), h_agl)
    n_h = haps_xyz_lonlatalt.shape[0]
    h_az, h_el, h_rng = [], [], []
    for i in range(n_h):
        lon_h, lat_h, alt_h = haps_xyz_lonlatalt[i]
        x, y, z = pm.geodetic2ecef(lat_h, lon_h, alt_h)
        az, el, rng = pm.ecef2aer(x, y, z, lat0, lon0, alt0)
        h_az.append(np.asarray(az, dtype=np.float64))
        h_el.append(np.asarray(el, dtype=np.float64))
        h_rng.append(np.asarray(rng, dtype=np.float64))
    haz = np.column_stack(h_az); hel = np.column_stack(h_el); hrng = np.column_stack(h_rng)
    habove = hel >= prep["mask_deg"]
    rec_i, sv_i = np.where(habove)
    hblocked = occlusion_mask(scene, prep["origins"][rec_i],
                              _enu_dirs(haz[rec_i, sv_i], hel[rec_i, sv_i]),
                              hrng[rec_i, sv_i], prep["max_batch"])
    hlos = np.zeros_like(habove)
    hlos[rec_i[~hblocked], sv_i[~hblocked]] = True
    los_aug = np.concatenate([prep["los"], hlos], axis=1)
    az_aug = np.concatenate([prep["azm"], haz], axis=1)
    el_aug = np.concatenate([prep["elm"], hel], axis=1)
    sig_aug = np.concatenate([prep["sigv"], np.full(n_h, haps_sig)])
    return _assemble_frame(prep, _metrics_arrays(
        prep, los_aug, az_aug, el_aug, sig_aug, confidence))


def evaluate_epoch(epoch_utc: pd.Timestamp, sv_geom: dict, recs: pd.DataFrame,
                   scene, cfg: dict, frame=None,
                   haps_xyz_lonlatalt: np.ndarray | None = None) -> pd.DataFrame:
    """Per-receiver availability + DOP + CRLB/PEB at one epoch (convenience
    wrapper: prepare + evaluate). See prepare_epoch / evaluate_prepared."""
    prep = prepare_epoch(epoch_utc, sv_geom, recs, scene, cfg)
    return evaluate_prepared(prep, cfg, haps_xyz_lonlatalt=haps_xyz_lonlatalt,
                             scene=scene)


def summarize_epoch(df: pd.DataFrame, q: tuple = (0.5, 0.9, 0.95)) -> dict:
    """Per-epoch availability statistics and DOP/PEB percentiles."""
    row = {"epoch": df["epoch"].iloc[0] if len(df) else None,
           "n_receivers": int(len(df)),
           "n_receivers_defined": int(df["hdop"].notna().sum()),
           "mean_n_above_mask": float(df["n_above_mask"].mean()) if len(df) else None,
           "mean_n_los": float(df["n_los"].mean()) if len(df) else None,
           "availability_mean": float(df["availability"].mean()) if len(df) else None}
    for col in ("hdop", "vdop", "pdop", "gdop", "tdop", "peb95_m"):
        s = df[col].dropna()
        for p in q:
            row[f"{col}_p{int(p * 100)}"] = float(s.quantile(p)) if len(s) else None
    return row


def pooled_metric(dfs: list[pd.DataFrame]) -> dict:
    """One scalar per metric over all epochs (documented aggregation):
    availability = pooled mean; hdop/vdop/peb95 = pooled median."""
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if all_df.empty:
        return {"availability": None, "hdop": None, "vdop": None, "peb95": None}
    return {
        "availability": float(all_df["availability"].mean()),
        "hdop": float(all_df["hdop"].median()),
        "vdop": float(all_df["vdop"].median()),
        "peb95": float(all_df["peb95_m"].median()),
    }
