# -*- coding: utf-8 -*-
"""HAPS-PNT: DOP / CRLB / PEB math (pure numpy, no I/O)."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2


def geometry_matrix(elev_deg: np.ndarray, azim_deg: np.ndarray) -> np.ndarray:
    """G matrix in ENU: rows [cos el cos az, cos el sin az, sin el, 1]."""
    el = np.radians(np.asarray(elev_deg, dtype=float))
    az = np.radians(np.asarray(azim_deg, dtype=float))
    n = el.size
    G = np.ones((n, 4))
    G[:, 0] = np.cos(el) * np.cos(az)
    G[:, 1] = np.cos(el) * np.sin(az)
    G[:, 2] = np.sin(el)
    return G


def dop_metrics(g_mat: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """HDOP / VDOP / PDOP / GDOP / TDOP from weighted GᵀG.

    weights: per-satellite variance weights (w = 1/sigma^2); None = unweighted.
    """
    g = np.asarray(g_mat, dtype=float)
    if g.ndim != 2 or g.shape[1] != 4:
        raise ValueError(f"G must be (n,4), got {g.shape}")
    if weights is None:
        w = np.ones(g.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != g.shape[0] or np.any(w <= 0):
            raise ValueError("weights must be positive and match G rows")
    wsqrt = np.sqrt(w)[:, None]
    h = g * wsqrt
    try:
        cov = np.linalg.inv(h.T @ h)
    except np.linalg.LinAlgError:
        return {"hdop": np.nan, "vdop": np.nan, "pdop": np.nan,
                "gdop": np.nan, "tdop": np.nan, "n_visible": g.shape[0]}
    hdop = math_sqrt(cov[0, 0] + cov[1, 1])
    vdop = math_sqrt(cov[2, 2])
    pdop = math_sqrt(cov[0, 0] + cov[1, 1] + cov[2, 2])
    gdop = math_sqrt(np.trace(cov))
    tdop = math_sqrt(cov[3, 3])
    return {"hdop": hdop, "vdop": vdop, "pdop": pdop,
            "gdop": gdop, "tdop": tdop, "n_visible": int(g.shape[0])}


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(float(x), 0.0)))


def crlb_cov(elev_deg: np.ndarray, azim_deg: np.ndarray,
             sigma_ure_m: float | np.ndarray, sigma_clock_m: float = 0.0) -> dict:
    """Full weighted CRLB covariance (m^2) of the LS position/clock estimate.

    Cov = (G^T W G)^-1,  W = diag( sin(el_i)^2 / sigma_i^2 ),
    with the fixed elevation-dependent UERE model sigma_i(el) = sigma_ure_i/sin(el)
    (el clipped to the valid geometric range (0, 90] deg); sigma_ure_i is the
    per-satellite URE scale (scalar broadcast, or one value per satellite,
    e.g. RINEX URA / Galileo SISA).

    The horizontal CRLB block is read directly from the covariance matrix:
        var_e = Cov[0,0],  var_n = Cov[1,1],  cov_en = Cov[0,1].
    No diagonal-only approximation is used.

    sigma_clock_m is accepted for API stability; the clock column of G already
    carries the common receiver-clock unknown, so it is not used here.
    """
    g = geometry_matrix(elev_deg, azim_deg)
    n = g.shape[0]
    el = np.clip(np.radians(np.asarray(elev_deg, dtype=float)), 1e-6, np.pi / 2)
    sigma = np.broadcast_to(np.asarray(sigma_ure_m, dtype=float), (n,)).copy()
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma_ure_m must be finite and > 0")
    # Cov = (G^T W G)^-1 with W = diag(sin^2 el / sigma_i^2)  [m^-2 weights]
    w = np.sin(el) ** 2 / sigma ** 2
    m = dop_metrics(g, weights=w)
    try:
        cov = np.linalg.inv((g * np.sqrt(w)[:, None]).T @ (g * np.sqrt(w)[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.nan)
    return {
        "cov_4x4": cov,
        "var_e_m2": float(cov[0, 0]),
        "var_n_m2": float(cov[1, 1]),
        "cov_en_m2": float(cov[0, 1]),
        "var_u_m2": float(cov[2, 2]),
        "cov_m2": {"e": float(cov[0, 0]), "n": float(cov[1, 1]), "u": float(cov[2, 2])},
        "hdop": m["hdop"], "vdop": m["vdop"], "pdop": m["pdop"],
        "gdop": m["gdop"], "tdop": m["tdop"], "n_visible": n,
    }


def peb_percentile(var_e_m2: float, var_n_m2: float, cov_en_m2: float = 0.0,
                   confidence: float = 0.95) -> float:
    """Horizontal position error bound (m) from the full 2x2 horizontal CRLB covariance.

    Published experiment protocol definition (experiment/experiment_config.json, crlb.peb_definition):
        PEB = sqrt(chi2.ppf(confidence, dof=2)) * sqrt(max_eigenvalue_of_horizontal_CRLB_cov)
    with C_h = [[var_e, cov_en], [cov_en, var_n]] (east-north covariance included).
    """
    k = float(chi2.ppf(confidence, df=2))       # = 5.991 for 95 %
    a, c, b = float(var_e_m2), float(var_n_m2), float(cov_en_m2)
    # max eigenvalue of the symmetric 2x2 matrix (closed form)
    lam = 0.5 * (a + c) + math.sqrt(0.25 * (a - c) ** 2 + b * b)
    lam = max(lam, 0.0)
    return float(np.sqrt(k * lam))
