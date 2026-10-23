# Experimental Input Data

This directory contains the input datasets used in the ACDSA 2027 study *Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments*.

## GNSS data

- `brdc/BRDC_normalized_GPS_GAL.rnx` — GPS and Galileo broadcast navigation data.
- `ephemeris/sv_table.csv` — Satellite ephemeris and navigation parameter table.

## Urban building geometry

OpenStreetMap-derived building footprints are provided for the three study regions:

- `shinjuku_tokyo/footprints.gpkg`
- `downtown_dubai/footprints.gpkg`
- `frankfurt_bankenviertel/footprints.gpkg`

Each city directory includes an `osm_provenance.json` file documenting the corresponding building dataset.

## Input verification

The original datasets are identified by their recorded SHA-256 checksums.

The input manifest is provided in `CAMERA_READY_INPUT_MANIFEST.json`. Experimental parameters and GNSS input checksums are specified in `../config/experiment.json`.

To verify the required input files and scientific source dependencies, run from the repository root:

```bash
python run/verify_inputs.py
```

The supplied input files correspond to the datasets used for the published numerical experiment. Reacquiring live OpenStreetMap data may produce different building geometries.

## Data attribution and licensing

Building geometry is derived from OpenStreetMap © OpenStreetMap contributors and is subject to the Open Database License (ODbL 1.0).

The GNSS navigation data have separate provenance and redistribution considerations.

See `../docs/DATA_PROVENANCE.md` for dataset origins, processing assumptions, attribution, and redistribution conditions.

Public distribution of third-party datasets is subject to confirmation of the applicable rights and obligations.