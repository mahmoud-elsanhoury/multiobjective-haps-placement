# -*- coding: utf-8 -*-
"""Step 1 of the published experiment protocol: OSM footprint extraction -> 3D geometry.

Provenance of scientific parameters (all fixed in experiment/experiment_config.json,
none invented here):
- ROI bboxes: fixed config `rois[*].bbox`, order [lat_min, lon_min, lat_max,
  lon_max] (WGS84), per the fixed bbox_note.
- Heights: OSM tagged `height` / `building:height` first; missing heights are
  imputed from the OSM-standard `building:levels` tag as levels * 3.0 m, the
  fixed convention `rois[*].height_imputation_levels` (levels_missing_m:
  levels*3.0). Imputation is reported, never hidden. Buildings with neither a
  usable height tag nor building:levels get NO height assigned (no fallback is
  invented); they are excluded from the 3D obstruction set and counted.
- Receiver grid: config `grid.*`.
Units: WGS-84 degrees in, local UTM meters out (pyproj, established library).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import trimesh

log = logging.getLogger("haps_pnt.geometry")

# fixed imputation convention: metres per building level
LEVELS_TO_M = 3.0

_HEIGHT_TAG_KEYS = ("height", "building:height")
_LEVELS_TAG_KEY = "building:levels"
_NUM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*m?\s*$")


# --------------------------------------------------------------------------- #
# Height parsing                                                              #
# --------------------------------------------------------------------------- #
def _tag_to_float(value):
    """Parse an OSM tag value to a positive finite float, else None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        m = _NUM_RE.match(value)
        if not m:
            return None
        v = float(m.group(1))
    else:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
    if math.isfinite(v) and v > 0:
        return v
    return None


def parse_height_m(value, *, height_scale: float = 1.0, level_shift: int = 0,
                   levels_to_m: float = LEVELS_TO_M):
    """Height in metres from an OSM building row (fixed rules only).

    Order: tagged height (`height` or `building:height`) ->
           imputed height from the OSM-standard `building:levels` tag:
           (levels + level_shift) * levels_to_m, kept only if > 0.
    `height_scale` (sensitivity factor) multiplies the resulting height.
    Returns (height_m_or_None, imputed: bool). No other fallback exists:
    rows without a usable tagged height and without building:levels yield None.
    """
    for key in _HEIGHT_TAG_KEYS:
        h = _tag_to_float(value.get(key))
        if h is not None:
            return h * height_scale, False
    lv = _tag_to_float(value.get(_LEVELS_TAG_KEY))
    if lv is not None:
        lv_eff = lv + int(level_shift)
        if lv_eff > 0:
            return lv_eff * levels_to_m * height_scale, True
    return None, True


def height_imputation_rate(rows, *, height_scale: float = 1.0, level_shift: int = 0) -> dict:
    """Data-quality covariate over building rows.

    Accepts either a GeoDataFrame (pipeline path) or a plain sequence of
    row dicts (unit-test path). Counting logic is identical for both.
    `n_no_height` counts rows with no usable height under the fixed rules
    (excluded from the 3D obstruction set; reported, never hidden).
    """
    if hasattr(rows, "apply"):            # GeoDataFrame-like
        parsed = rows.apply(
            lambda r: parse_height_m(r.to_dict(), height_scale=height_scale,
                                     level_shift=level_shift), axis=1)
    else:                                  # plain sequence of dicts
        parsed = [parse_height_m(dict(r), height_scale=height_scale,
                                 level_shift=level_shift) for r in rows]
    heights = [h for h, _ in parsed]
    imputed_flags = [imp and h is not None for h, imp in parsed]

    n = len(heights)
    if n == 0:
        return {"n_buildings": 0, "n_imputed": 0, "imputation_rate": None,
                "n_no_height": 0, "height_min_m": None, "height_max_m": None}
    n_imputed = int(sum(imputed_flags))
    usable = [h for h in heights if h is not None]
    return {
        "n_buildings": int(n),
        "n_imputed": n_imputed,
        "imputation_rate": n_imputed / n,
        "n_no_height": int(n - len(usable)),
        "height_min_m": float(np.nanmin(usable)) if usable else None,
        "height_max_m": float(np.nanmax(usable)) if usable else None,
    }


# --------------------------------------------------------------------------- #
# Local metric projection                                                     #
# --------------------------------------------------------------------------- #
def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    # WGS-84 / UTM: EPSG 326xx (north) / 327xx (south), zone = 1..60
    return (32600 if lat >= 0 else 32700) + zone


def roi_center(roi: dict) -> tuple[float, float]:
    """True center (lon, lat) of a fixed ROI bbox [lat_min, lon_min, lat_max, lon_max]."""
    lat_min, lon_min, lat_max, lon_max = roi["bbox"]
    return (lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0


class LocalFrame:
    """WGS-84 lon/lat <-> local tangent-plane metres (pyproj, established)."""

    def __init__(self, lon0: float, lat0: float):
        from pyproj import CRS, Transformer
        self.crs = CRS.from_epsg(_utm_epsg(lon0, lat0))
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def to_local(self, lon, lat):
        return self._fwd.transform(lon, lat)

    def to_wgs84(self, x, y):
        return self._inv.transform(x, y)


# --------------------------------------------------------------------------- #
# Mesh building                                                               #
# --------------------------------------------------------------------------- #
def footprints_to_mesh(gdf: gpd.GeoDataFrame, frame: LocalFrame, *,
                       height_scale: float = 1.0, level_shift: int = 0) -> tuple:
    """Extrude every footprint polygon by its (possibly imputed) height.

    Polygons are projected to local metres first so that extrusion heights
    are true metres; ray tracing then operates in a consistent unit system.
    Rows with no usable height under the fixed rules (or non-polygon
    geometries) are excluded from the mesh and counted; nothing is invented.
    Returns (mesh, {"n_excluded_no_height", "n_excluded_geometry"}).
    """
    meshes = []
    n_excl_height = 0
    n_excl_geom = 0
    from shapely.ops import transform as shp_transform
    for _, row in gdf.iterrows():
        h, imputed = parse_height_m(row.to_dict(), height_scale=height_scale,
                                    level_shift=level_shift)
        if h is None:
            n_excl_height += 1
            continue
        geom = row.geometry
        if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon"):
            n_excl_geom += 1
            continue
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in parts:
            try:
                poly_local = shp_transform(frame.to_local, poly)
                m = trimesh.creation.extrude_polygon(poly_local, height=float(h))
            except Exception as exc:  # keep going: preserve failed rows in report
                log.warning("extrusion failed (%s); row preserved in report", exc)
                continue
            m.metadata["building_height_m"] = float(h)
            m.metadata["imputed"] = bool(imputed)
            meshes.append(m)
    if not meshes:
        raise RuntimeError("no buildable footprints in ROI")
    return trimesh.util.concatenate(meshes), {
        "n_excluded_no_height": n_excl_height,
        "n_excluded_geometry": n_excl_geom,
    }


# --------------------------------------------------------------------------- #
# Run entry point (step 1)                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class GeometryResult:
    roi_id: str
    gpkg_path: Path
    glb_path: Path
    gpkg_sha256: str
    glb_sha256: str
    stats: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_geometry_stage(cfg: dict, out_dir: Path,
                       input_gpkg: dict | None = None,
                       source_labels: dict | None = None) -> list[GeometryResult]:
    """Step 1 per published experiment protocol.

    input_gpkg: optional {roi_id: path} of preserved OSM footprint exports
    (GPKG, EPSG:4326). When present the preserved input is reused and OSM is
    NOT refetched; otherwise footprints are fetched from Overpass per config.
    Fresh acquisitions follow the fixed data policy (300 s timeout) and store
    reproducible provenance + the raw Overpass response per ROI.
    source_labels: optional {roi_id: label} overriding the recorded source.
    """
    import osmnx as ox

    ox.settings.requests_timeout = 300  # fixed OSM data policy: 300 s timeout
    ox.settings.use_cache = True
    cache_dir = Path(out_dir) / "osm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(cache_dir)

    heights_cfg = cfg.get("heights") or {}
    scale = float(heights_cfg.get("scale", 1.0))
    shift = int(heights_cfg.get("level_shift", 0))
    results = []
    for roi in cfg["rois"]:
        # fixed bbox order: [lat_min, lon_min, lat_max, lon_max]
        lat_min, lon_min, lat_max, lon_max = roi["bbox"]
        center_lon, center_lat = (lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0
        frame = LocalFrame(center_lon, center_lat)
        roi_dir = out_dir / roi["id"]
        roi_dir.mkdir(parents=True, exist_ok=True)

        provenance: dict = {
            "roi_id": roi["id"],
            "frozen_bbox_lat_min_lon_min_lat_max_lon_max": roi["bbox"],
            "tags": {"building": True},
            "used_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        preserved = (input_gpkg or {}).get(roi["id"])
        if preserved is not None and Path(preserved).exists():
            preserved = Path(preserved)
            log.info("[%s] reusing preserved OSM input %s (no refetch)", roi["id"], preserved)
            gdf = gpd.read_file(preserved)
            label = (source_labels or {}).get(roi["id"])
            source = label or f"preserved:{preserved.name}"
            provenance.update({
                "source": source,
                "osm_refetch": False,
                "input_path": str(preserved),
                "input_sha256": sha256_file(preserved),
            })
        else:
            log.info("[%s] downloading OSM footprints", roi["id"])
            cache_before = {p.name for p in cache_dir.glob("*")}
            t0 = datetime.now(timezone.utc)
            gdf = ox.features.features_from_bbox(
                bbox=(lon_min, lat_min, lon_max, lat_max),  # osmnx 2.x: (left, bottom, right, top)
                tags={"building": True},
            )
            t1 = datetime.now(timezone.utc)
            raw_files = {}
            for p in sorted(cache_dir.glob("*")):
                if p.name not in cache_before:
                    raw_files[p.name] = {"sha256": sha256_file(p), "n_bytes": p.stat().st_size}
            provenance.update({
                "source": "overpass",
                "osm_refetch": True,
                "osmnx_version": ox.__version__,
                "overpass_url": getattr(ox.settings, "overpass_url", None),
                "overpass_settings": str(getattr(ox.settings, "overpass_settings", None)),
                "requests_timeout_s": int(getattr(ox.settings, "requests_timeout", 0) or 0),
                "request_utc": t0.isoformat(timespec="seconds"),
                "response_utc": t1.isoformat(timespec="seconds"),
                "raw_response_cache_dir": str(cache_dir),
                "raw_response_files": raw_files,
                "n_elements_returned": int(len(gdf)),
                "geometry_type_counts": {
                    k: int(v) for k, v in gdf.geometry.geom_type.value_counts().items()},
            })
            source = "overpass"
        gdf = gdf.reset_index(drop=True)
        stats = height_imputation_rate(gdf, height_scale=scale, level_shift=shift)
        log.info(
            "[%s] %d buildings, imputation rate %.1f%%, no-height %d",
            roi["id"], stats["n_buildings"], 100 * (stats["imputation_rate"] or 0.0),
            stats["n_no_height"],
        )
        mesh, excl = footprints_to_mesh(gdf, frame, height_scale=scale, level_shift=shift)
        gpkg = roi_dir / "footprints.gpkg"
        gdf.to_file(gpkg, driver="GPKG")
        glb = roi_dir / "buildings.glb"
        scene = trimesh.Scene({"buildings": mesh})
        scene.export(glb)
        provenance["output_gpkg_sha256"] = sha256_file(gpkg)
        provenance["output_glb_sha256"] = sha256_file(glb)
        (roi_dir / "osm_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str), encoding="utf-8")
        results.append(
            GeometryResult(
                roi_id=roi["id"],
                gpkg_path=gpkg,
                glb_path=glb,
                gpkg_sha256=sha256_file(gpkg),
                glb_sha256=sha256_file(glb),
                stats={**stats, **excl, "crs": str(frame.crs),
                       "center_lon": center_lon, "center_lat": center_lat,
                       "height_scale": scale, "level_shift": shift,
                       "osm_source": source},
            )
        )
    return results
