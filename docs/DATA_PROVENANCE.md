# Scientific Data Provenance

This document describes the datasets used in *Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments* (ACDSA 2027).

The repository includes the experimental input files used for the reported numerical results. Their integrity can be checked using `run/verify_inputs.py`.

## 1. Urban building geometry

Building footprints were obtained from **OpenStreetMap (OSM)** using OSMnx 2.1.1 and the Overpass API.

Three urban regions were considered:

| Region | Acquisition (UTC) | Building polygons |
|---|---|---:|
| Shinjuku, Tokyo | 2026-10-13 16:45:01 | 10,592 |
| Downtown Dubai | 2026-10-13 16:46:05 | 864 |
| Frankfurt Bankenviertel | 2026-10-13 16:46:12 | 3,743 |

The geographical boundaries are specified in `config/experiment.json`.

Building heights were obtained from tagged height attributes when available. Otherwise, the `building:levels` attribute was converted using an assumed floor height of 3 m. Buildings without usable height information were excluded from the three-dimensional obstruction model.

The datasets are stored in:

- `data/shinjuku_tokyo/footprints.gpkg`
- `data/downtown_dubai/footprints.gpkg`
- `data/frankfurt_bankenviertel/footprints.gpkg`

Each region includes an `osm_provenance.json` file recording its acquisition timestamp, geographical extent, feature counts, height-processing information, and SHA-256 checksum.

Additional dataset metadata are available in `data/CAMERA_READY_INPUT_MANIFEST.json` and `data/new_city_data_audit.csv`.

The included GeoPackages are the fixed experimental datasets. Repeating a live OpenStreetMap query may return different data as the underlying database evolves.

## 2. GNSS navigation data

The experiment uses GPS and Galileo broadcast navigation information for 30 August 2026.

The required files are:

| File | Description |
|---|---|
| `data/brdc/BRDC_normalized_GPS_GAL.rnx` | RINEX 3.05 broadcast navigation data |
| `data/ephemeris/sv_table.csv` | Satellite orbital and navigation parameter table |

The RINEX header identifies BKG Frankfurt and records processing with Converto and gfzrnx. These header entries do not, by themselves, establish the original download location or redistribution license.

The complete experiment uses 25 epochs from 11:00 to 13:00 UTC, separated into 18 DESIGN epochs and 7 held-out TEST epochs.

The exact input-file SHA-256 checksums are specified in `config/experiment.json`.

## 3. Input integrity

The experiment uses fixed input datasets to ensure consistent numerical evaluation.

Before execution, run:

```bash
python run/verify_inputs.py
```

The verification utility checks the required experimental files against their recorded checksums.

The original numerical results can be compared with the contents of `reference_results/` using `analysis/validate_results.py`.

## 4. Attribution and redistribution

**OpenStreetMap building data**

Building geometry is derived from OpenStreetMap:

© OpenStreetMap contributors

OpenStreetMap data is licensed under the Open Data Commons Open Database License (ODbL 1.0).

- https://www.openstreetmap.org/copyright
- https://opendatacommons.org/licenses/odbl/1-0/

Redistribution and adaptation of OSM-derived databases must comply with the applicable ODbL attribution and share-alike provisions.

**GNSS navigation data**

The RINEX header identifies BKG Frankfurt and the processing software, but the exact download archive was not recorded.



The International GNSS Service (IGS) provides openly accessible GNSS navigation data and products and requests appropriate acknowledgment of IGS and contributing organizations.



Reference: https://igs.org/about-2/



The RINEX file and associated satellite table are scientific inputs, not MIT-licensed software. The exact normalized file and its original checksums are preserved. Redistribution conditions applicable to this specific copy should be confirmed before public distribution.

**Software**

The repository's software license is separate from the licensing conditions governing the scientific datasets.

## 5. Reproducibility considerations

The original data files and their checksums define the experimental input state.

Results depend on the documented building geometry, GNSS broadcast navigation information, experimental configuration, and scientific implementation.

The study uses geometric line-of-sight analysis and positioning uncertainty bounds. Limitations associated with building-height completeness and propagation modeling are discussed in the associated paper.