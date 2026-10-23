# Reference Results

This directory contains the reference results and supporting evidence for *Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments* (ACDSA 2027).

These files provide a basis for checking independently reproduced results against the reported experiment.

## 1. Scientific results

The `raw/` directory contains the 13 original top-level result files, preserving their numerical values and experimental selections.

These include:

- DESIGN-selected HAPS layouts and representative solutions.
- Recovery and positioning error bound (PEB95) results.
- Structured placement and refinement results.
- Seed-level TEST statistics.
- Common-population positioning quality data.
- Urban geometry summaries.

## 2. Receiver-level TEST evidence

The `detail_evidence/` directory contains 27 receiver-level TEST result files covering the nine reported placement cases in each of the three cities.

These files support verification of recovery rates and descriptive PEB95 statistics.

For reporting purposes, recovered cases with non-finite PEB95 values are retained in recovery counts but excluded from the descriptive median PEB95 calculation. The optimization objectives and selection rules remain unchanged.

## 3. Publication reporting audit

The `publication_audit/` directory contains:

- `camera_ready_reporting_table_with_peb_audit.csv`
- `reporting_detail_verification.csv`

These files provide numerical checks of the publication results using the receiver-level TEST evidence.

The reporting audit is derived from the original experimental outputs without modifying them.

## 4. Manuscript tables

The `tables/` directory contains the regenerated publication tables, including the main results, DESIGN-selected representatives, recovery improvements, and seed variability.

The tables are generated from the scientific results using `analysis/build_tables.py`.

## 5. Figures

The `figures/` directory contains visualizations generated from the reference results, including HAPS placement layouts, DESIGN Pareto comparisons, common-population positioning quality, and seed variability.

These figures are supporting reproductions and are not asserted to be byte-identical to the manuscript's original graphics.

The corresponding plotting scripts are provided under `analysis/`.

## 6. Numerical verification

The reproduced results can be compared against the supplied reference files using:

```bash
python analysis/validate_results.py
```

An independent numerical reproduction passed the five reference-result comparisons within the validation script's absolute and relative tolerances of `1e-8`.

For the execution procedure and experimental limitations, see `docs/REPRODUCIBILITY.md`.