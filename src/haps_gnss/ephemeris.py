# -*- coding: utf-8 -*-
"""Step 2 of the published experiment protocol: public ephemerides -> per-epoch satellite geometry.

Data source (verified 2026-08-31 by HEAD request, status 200):
    IGS/BKG daily merged broadcast ephemeris (BRDC), RINEX 3:
    https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy}/BRDC00WRD_R_{year}{doy}0000_01D_MN.rnx.gz
No other mirror is used (no fabricated fallbacks); if the verified endpoint
fails the stage records the failure and stops.

Processing chain:
    BRDC RINEX --layout normalization (GPS/Galileo)--> georinex
        --Keplerian elements --IS-GPS-200 propagation--> ECEF (m)
    ECEF --pymap3d--> ENU per receiver --> DOP / CRLB inputs

SV-table fidelity (ephemeris-record fidelity)
--------------------------------------------
georinex 1.16.2 fixes the field layout of each SV from its FIRST record and
then silently blanks every record whose layout differs (padded short records
fail float-parsing on blank-only trailing fields; short records parsed under a
long layout misassign fields from the spare positions). The preserved BRDC
file mixes layouts per constellation (GPS: 29/31 fields; Galileo: 27/28/31
fields), which caused silent record loss. Fix applied here (least invasive,
standards-compliant, deterministic):

  * the original BRDC file is never modified (sha256 pinned + verified);
  * a DERIVED normalized copy is produced in which every GPS record has the
    canonical 29 fields (RINEX 3.05 Table A23) and every Galileo record the
    canonical 28 fields (Table A25); trailing spare fields are dropped only
    after verifying they are blank/zero, and the Galileo spare0 field is
    inserted as 0.0 where a source record omits it (spare fields carry no
    information);
  * all other constellations and the header are copied verbatim;
  * the derived file is recorded as a reproducibility artifact with its own
    provenance (source sha256, transformation, counts, derived sha256).

Duplicate (sv, ToC) ephemerides (georinex phantom SV ids such as 'E02_1') are
merged back onto the parent SV id; exact-duplicate records are dropped and
distinct ephemerides are preserved, deterministically ordered.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import requests

import georinex as gr
import numpy as np
import pandas as pd
import pymap3d as pm
import xarray as xr

log = logging.getLogger("haps_pnt.ephemeris")

# WGS-84 / IS-GPS-200 constants (standard, not tuned)
MU_EARTH = 3.986005e14          # m^3/s^2, ICD-GPS-200
OMEGA_E_DOT = 7.2921151467e-5   # rad/s, ICD-GPS-200
HALF_WEEK = 302400.0            # s, week rollover

# fixed constellation scope: GPS + Galileo only
SCOPE_SYSTEMS = ("G", "E")

# canonical RINEX 3.05 field counts per system (broadcast orbit records)
CANONICAL_N = {"G": 29, "E": 28}
# accepted source layouts (observed in the preserved BRDC product)
ACCEPTED_N = {"G": (29, 31), "E": (27, 28, 31)}

LF = 19  # RINEX 3 field width
NLINES = {"C": 7, "E": 7, "G": 7, "J": 7, "R": 3, "S": 3, "I": 7}


# --------------------------------------------------------------------------- #
# 1. Download with provenance (first-run path; resume reuses preserved input) #
# --------------------------------------------------------------------------- #
def brdc_url(year: int, doy: int) -> str:
    return (
        "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/"
        f"{year}/{doy:03d}/BRDC00WRD_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
    )


def download_brdc(year: int, doy: int, out_path: Path, timeout_s: int = 300) -> dict:
    """Download the fixed BRDC file; record identity + sha256. No silent retry."""
    url = brdc_url(year, doy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
    prov = {
        "source_url": url,
        "status": "downloaded",
        "bytes": size,
        "sha256": h.hexdigest(),
        "filename": out_path.name,
    }
    (out_path.parent / "PROVENANCE.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    log.info("BRDC downloaded: %s (%d bytes, sha256 %s…)", out_path.name, size, prov["sha256"][:12])
    return prov


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_brdc_source(path: Path, expected_sha256: str | None) -> dict:
    """Verify the preserved BRDC input is intact before any reuse/derivation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"preserved BRDC input missing: {path}")
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256.lower():
        raise ValueError(
            f"BRDC integrity check FAILED for {path.name}: "
            f"expected sha256 {expected_sha256[:16]}…, got {actual[:16]}…"
        )
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size, "verified": True}


# --------------------------------------------------------------------------- #
# 2. Deterministic layout normalization (derived reproducibility artifact)    #
# --------------------------------------------------------------------------- #
def _is_epoch_line(ln: str) -> bool:
    if len(ln) < 23 or ln[0] not in NLINES:
        return False
    if not (ln[1:3].strip().isdigit()):
        return False
    try:
        int(ln[4:8]); int(ln[9:11]); int(ln[12:14])
        int(ln[15:17]); int(ln[18:20]); int(ln[21:23])
        return True
    except ValueError:
        return False


def _spare_is_empty(field: str) -> bool:
    """True if a spare field is blank or numerically zero (no information)."""
    s = field.strip()
    if not s:
        return True
    try:
        return float(s.replace("D", "E")) == 0.0
    except ValueError:
        return False


def normalize_brdc(src: Path, dst: Path, prov_path: Path | None = None,
                   expected_src_sha256: str | None = None) -> dict:
    """Normalize GPS/Galileo broadcast-record layouts to the canonical RINEX 3.05
    field counts (GPS 29, Galileo 28). Deterministic; original never modified.

    Returns provenance/stats dict; writes `prov_path` (default: dst's dir
    NORMALIZATION_PROVENANCE.json). Raises loudly on any unsupported layout or
    on a non-empty spare field that would have to be dropped.
    """
    src, dst = Path(src), Path(dst)
    src_prov = verify_brdc_source(src, expected_src_sha256)

    if src.suffix == ".gz":
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    else:
        lines = src.read_text(encoding="utf-8").splitlines()

    try:
        hdr_end = next(i for i, l in enumerate(lines) if "END OF HEADER" in l)
    except StopIteration:
        raise ValueError("no END OF HEADER found; not a RINEX NAV file")

    out: list[str] = list(lines[: hdr_end + 1])
    counts = {"G": {}, "E": {}}
    n_trimmed = {"G": 0, "E": 0}
    n_padded = {"E": 0}
    n_copied_other = 0

    i = hdr_end + 1
    while i < len(lines):
        ln = lines[i]
        if not _is_epoch_line(ln):
            if ln.strip():
                out.append(ln)  # preserve unexpected non-empty lines verbatim
            i += 1
            continue
        sysc = ln[0]
        n_extra = NLINES[sysc]
        block = lines[i: i + 1 + n_extra]
        if len(block) < 1 + n_extra:
            raise ValueError(f"truncated ephemeris block at line {i + 1} ({ln[:3]})")
        if sysc not in SCOPE_SYSTEMS:
            out.extend(block)          # out-of-scope systems: copied verbatim
            n_copied_other += 1
            i += 1 + n_extra
            continue

        n_canon = CANONICAL_N[sysc]
        raw = ln[23:80]
        for extra in block[1:]:
            raw += extra[4:80]
        n_src = math.ceil(len(raw.rstrip()) / LF) if raw.rstrip() else 0
        counts[sysc][n_src] = counts[sysc].get(n_src, 0) + 1
        if n_src not in ACCEPTED_N[sysc]:
            raise ValueError(
                f"unsupported {sysc} record layout: N={n_src} fields at line {i + 1}; "
                f"accepted: {ACCEPTED_N[sysc]} (no silent loss allowed)"
            )
        fields = [raw[LF * k: LF * (k + 1)] for k in range(n_src)]

        if sysc == "G":
            if n_src == 31:  # standard 29 fields + trailing spare0/spare1
                for k, spare in ((29, fields[29]), (30, fields[30])):
                    if not _spare_is_empty(spare):
                        raise ValueError(
                            f"GPS record at line {i + 1}: trailing spare field {k} "
                            f"is non-empty ({spare!r}); refusing to drop information"
                        )
                fields = fields[:29]
                n_trimmed["G"] += 1
        else:  # E
            if n_src == 31:  # canonical 28 + trailing spare1/spare2/spare3
                for k in (28, 29, 30):
                    if not _spare_is_empty(fields[k]):
                        raise ValueError(
                            f"Galileo record at line {i + 1}: trailing spare field {k} "
                            f"is non-empty ({fields[k]!r}); refusing to drop information"
                        )
                fields = fields[:28]
                n_trimmed["E"] += 1
            elif n_src == 27:  # record omits the spare0 field (index 22)
                fields = fields[:22] + [" 0.000000000000e+00"] + fields[22:27]
                n_padded["E"] += 1
        assert len(fields) == n_canon

        out.append(ln.rstrip())  # epoch line (sv + ToC + clock) unchanged
        rest = fields[3:]
        for j in range(0, len(rest), 4):
            out.append((" " * 4 + "".join(f.ljust(LF) for f in rest[j: j + 4])).rstrip())
        i += 1 + n_extra

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    dst_sha = sha256_file(dst)

    prov = {
        "artifact": "derived_normalized_brdc",
        "source_file": src.name,
        "source_sha256": src_prov["sha256"],
        "source_bytes": src_prov["bytes"],
        "source_verified": True,
        "transformation": (
            "Per-constellation RINEX-3 NAV layout normalization only: "
            "GPS -> 29 fields (drop trailing spare0/spare1 after blank/zero check); "
            "Galileo -> 28 fields (drop trailing spare1..3 after blank/zero check; "
            "insert spare0=0.0 where omitted). Header and all other constellations "
            "copied verbatim. No ephemeris values altered."
        ),
        "scope_systems": list(SCOPE_SYSTEMS),
        "records_per_layout_in": {s: {str(k): v for k, v in sorted(d.items())}
                                  for s, d in counts.items()},
        "records_trimmed": n_trimmed,
        "records_padded_spare0": n_padded,
        "records_other_systems_copied": n_copied_other,
        "derived_sha256": dst_sha,
        "derived_file": dst.name,
        "original_file_modified": False,
    }
    prov_out = Path(prov_path) if prov_path else dst.parent / "NORMALIZATION_PROVENANCE.json"
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    log.info("BRDC normalized -> %s (in: %s, trimmed G=%d E=%d, padded E=%d)",
             dst.name, counts, n_trimmed["G"], n_trimmed["E"], n_padded["E"])
    return prov


# --------------------------------------------------------------------------- #
# 3. Kepler propagation (IS-GPS-200, single implementation, unit-tested)      #
# --------------------------------------------------------------------------- #
@dataclass
class SatState:
    svid: int
    toe: float          # epoch of ephemeris, s of week
    sqrt_a: float       # sqrt(m)
    ecc: float
    i0: float           # rad
    omega0: float       # right ascension of ascending node, rad
    arg_perigee: float  # rad
    m0: float           # rad
    delta_n: float      # rad/s
    cus: float
    cuc: float
    crs: float
    crc: float
    cis: float
    cic: float
    idot: float         # rad/s
    omega_dot: float    # rad/s


def kepler_ecc_anom(m: float, e: float, tol: float = 1e-13, itmax: int = 30) -> float:
    """Solve E - e sin E = M (Newton, standard)."""
    ecc = m if e < 0.8 else math.pi
    for _ in range(itmax):
        f = ecc - e * math.sin(ecc) - m
        fp = 1.0 - e * math.cos(ecc)
        d = -f / fp
        ecc += d
        if abs(d) < tol:
            break
    return ecc


def sv_position_ecef(sv: SatState, t: float, t_ref: float) -> np.ndarray:
    """IS-GPS-200 Table 20-IV (subframes 1-3) -> ECEF position (m)."""
    A = sv.sqrt_a ** 2
    n0 = math.sqrt(MU_EARTH / A ** 3)
    n = n0 + sv.delta_n
    tk = _tk(t, sv.toe)
    Mk = sv.m0 + n * tk
    Ek = kepler_ecc_anom(Mk, sv.ecc)
    sinE, cosE = math.sin(Ek), math.cos(Ek)
    nu = math.atan2(math.sqrt(1 - sv.ecc ** 2) * sinE, cosE - sv.ecc)
    phi = nu + sv.arg_perigee
    dphi = sv.cus * math.sin(2 * phi) + sv.cuc * math.cos(2 * phi)
    dr = sv.crs * math.sin(2 * phi) + sv.crc * math.cos(2 * phi)
    di = sv.cis * math.sin(2 * phi) + sv.cic * math.cos(2 * phi)
    u = phi + dphi
    r = A * (1 - sv.ecc * math.cos(Ek)) + dr
    i = sv.i0 + di + sv.idot * tk
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)
    om = sv.omega0 + (sv.omega_dot - OMEGA_E_DOT) * tk - OMEGA_E_DOT * sv.toe
    cos_o, sin_o = math.cos(om), math.sin(om)
    cos_i, sin_i = math.cos(i), math.sin(i)
    x = x_orb * cos_o - y_orb * cos_i * sin_o
    y = x_orb * sin_o + y_orb * cos_i * cos_o
    z = y_orb * sin_i
    return np.array([x, y, z])


def _tk(t: float, t_ref: float) -> float:
    dt = t - t_ref
    if dt > HALF_WEEK:
        dt -= 2 * HALF_WEEK
    elif dt < -HALF_WEEK:
        dt += 2 * HALF_WEEK
    return dt


# --------------------------------------------------------------------------- #
# 4. ECEF -> ENU per receiver (pymap3d, established)                          #
# --------------------------------------------------------------------------- #
def ecef_to_enu(sat_ecef: np.ndarray, rec: tuple[float, float, float]) -> tuple[float, float, float]:
    """az / el / slant range from receiver geodetic position (pymap3d)."""
    lon, lat, alt = pm.ecef2geodetic(sat_ecef[0], sat_ecef[1], sat_ecef[2])
    az, el, slant = pm.ecef2aer(
        sat_ecef[0], sat_ecef[1], sat_ecef[2],
        rec[0], rec[1], rec[2],
    )
    return az, el, slant


# --------------------------------------------------------------------------- #
# 5. Stage entry points                                                       #
# --------------------------------------------------------------------------- #
def load_brdc(path: Path, use: set[str] | None = None) -> xr.Dataset:
    """Parse (normalized) BRDC RINEX via georinex.

    `use` defaults to the fixed GPS+Galileo scope. Mixed files are returned
    as a single merged Dataset.
    """
    if use is None:
        use = set(SCOPE_SYSTEMS)
    ds = gr.load(path, use=use)
    if isinstance(ds, list):  # mixed constellation files
        ds = xr.merge(ds, combine_attrs="drop_conflicts")
    return ds


#: canonical broadcast-orbit aliases (canonical -> accepted variable names)
EPHEMERIS_ALIASES = {
    "sqrt_a":   ("sqrtA", "sqrt(A)"),
    "ecc":      ("ecc", "Eccentricity", "e"),
    "i0":       ("i0", "Io", "inclination"),
    "omega0":   ("OMEGA0", "Omega0", "longitude_of_ascending_node"),
    "arg_perigee": ("argPerigee", "omega", "argument_of_perigee"),
    "m0":       ("M0", "mean_anomaly"),
    "delta_n":  ("deltaN", "DeltaN"),
    "cus":      ("cus", "Cus"),
    "cuc":      ("cuc", "Cuc"),
    "crs":      ("crs", "Crs"),
    "crc":      ("crc", "Crc"),
    "cis":      ("cis", "Cis"),
    "cic":      ("cic", "Cic"),
    "idot":     ("idot", "IDOT", "iDot"),
    "omega_dot": ("OMEGA_DOT", "OmegaDot"),
    "toe":      ("Toe", "toe"),
    "sv_clock_bias": ("SVclockBias", "svclockbias"),
    "sv_clock_drift": ("SVclockDrift", "svclockdrift"),
    "sv_acc":   ("SVacc", "svacc", "URA"),
    "sisa":     ("SISA", "sisa"),
    "health":   ("health", "SatH1"),
}


def sv_table(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten ephemeris Dataset -> one row per present (sv, ToC) record.

    Fidelity rules (ephemeris-record fidelity):
    - georinex phantom duplicate-SV coordinates ('E02_1', …) are merged back
      onto the parent SV id ('E02'); they represent distinct ephemerides
      uploaded at the same ToC and are preserved as separate rows;
    - rows that are exact duplicates across ALL fields are dropped (counted);
      distinct ephemerides at the same ToC are never dropped;
    - only rows with finite sqrt_a are emitted (record actually present);
    - deterministic order: sv, toc, toe.
    """
    var_ci = {str(v).lower(): v for v in ds.data_vars}

    def resolve(canon: str):
        for alias in EPHEMERIS_ALIASES.get(canon, ()):
            v = var_ci.get(alias.lower())
            if v is not None:
                return v
        return None

    fields = {}
    for dst in EPHEMERIS_ALIASES:
        src = resolve(dst)
        if src is not None:
            fields[src] = dst
    if "sqrt_a" not in fields.values():
        raise ValueError("no broadcast-orbit ephemeris fields found in dataset")

    svs = [str(s) for s in np.atleast_1d(ds["sv"].values)]
    times = pd.to_datetime(np.atleast_1d(ds["time"].values))

    # presence marker
    sqrt_a_src = next(src for src, dst in fields.items() if dst == "sqrt_a")
    presence_var = ds[sqrt_a_src]

    data: dict[str, list] = {"sv": [], "toc": []}
    for dst in fields.values():
        data[dst] = []
    n_phantom_merged = 0
    for sv_idx, sv_coord in enumerate(svs):
        base_sv = sv_coord.split("_")[0]   # merge phantom duplicate copies
        if base_sv != sv_coord:
            n_phantom_merged += 1
        pres = np.isfinite(presence_var.values[:, sv_idx]) if presence_var.dims == ("time", "sv") \
            else np.isfinite(presence_var.values[sv_idx, :])
        idx = np.where(pres)[0]
        for t_i in idx:
            data["sv"].append(base_sv)
            data["toc"].append(times[t_i])
            for src, dst in fields.items():
                arr = ds[src].values
                if ds[src].dims == ("time", "sv"):
                    data[dst].append(arr[t_i, sv_idx])
                else:
                    data[dst].append(arr[sv_idx, t_i])
    df = pd.DataFrame(data)
    n_before = len(df)
    df = df.drop_duplicates()
    n_exact_dupes = n_before - len(df)
    df = df.sort_values(["sv", "toc", "toe"], ignore_index=True)
    df.attrs["n_phantom_sv_coords_merged"] = n_phantom_merged
    df.attrs["n_exact_duplicate_records_dropped"] = n_exact_dupes
    return df


def select_epoch(svdf: pd.DataFrame, epoch) -> pd.DataFrame:
    """Deterministic per-SV ephemeris selection at one propagation epoch.

    Rule (standard broadcast usage, no invented thresholds): for each SV use
    the most recent upload with ToC <= epoch; if no upload precedes the epoch,
    use the earliest upload and flag `toc_fallback=True` (reported, never
    hidden). Ties broken deterministically by earlier Toe.
    """
    epoch = pd.Timestamp(epoch)
    if epoch.tzinfo is not None:  # RINEX time tags are UTC-naive; compare naive
        epoch = epoch.tz_convert("UTC").tz_localize(None)
    rows = []
    for sv, grp in svdf.groupby("sv", sort=True):
        past = grp[grp["toc"] <= epoch]
        if len(past):
            best = past.sort_values(["toc", "toe"], ascending=[False, True]).iloc[0]
            fallback = False
        else:
            best = grp.sort_values(["toc", "toe"]).iloc[0]
            fallback = True
        rec = best.to_dict()
        rec["delta_toc_s"] = (epoch - rec["toc"]).total_seconds()
        rec["toc_fallback"] = fallback
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out


def propagate_epoch(sv: SatState, t_utc: float, rec: tuple[float, float, float]) -> dict:
    """Position at one epoch -> az / el / slant range from a receiver."""
    p = sv_position_ecef(sv, t_utc, sv.toe)
    az, el, rng = ecef_to_enu(p, rec)
    return {"az_deg": az, "el_deg": el, "range_m": rng}
