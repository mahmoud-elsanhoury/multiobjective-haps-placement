# Reproducibility Guide

This repository contains the scientific implementation, experimental configuration, input datasets, and reference results for the ACDSA 2027 study *Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments*.

## 1. Environment

The experiment uses Python 3.12 and the dependencies specified in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The complete experiment is computationally intensive and is best executed on a high-performance computing system with sufficient memory and processing capacity.

## 2. Scientific inputs

The `data/` directory contains GNSS broadcast navigation data, satellite ephemerides, and OpenStreetMap-derived building geometry for Shinjuku, Downtown Dubai, and Frankfurt Bankenviertel.

The `inputs` symbolic link points to `data` for compatibility with the experimental notebook.

Input checksums and experimental parameters are recorded in `config/experiment.json` and the input manifest.

Verify the required files before execution:

```bash
python run/verify_inputs.py
```

## 3. Experiment execution

Run the numerical checks without executing the complete optimization:

```bash
python run/reproduce.py --smoke
```

Execute the full experiment:

```bash
python run/reproduce.py
```

The experiment performs GNSS visibility analysis, HAPS placement optimization, DESIGN-based layout selection, and held-out TEST evaluation.

The experimental protocol uses 25 epochs on 30 August 2026, divided into 18 DESIGN epochs and 7 TEST epochs. All optimization, representative selection, and refinement decisions use DESIGN data exclusively.

Generated scientific results are stored under `outputs/reproduction_run/`. Publication tables and figures are generated under `publication_outputs/`.

## 4. Results validation

Compare reproduced scientific outputs with the supplied reference results:

```bash
python analysis/validate_results.py
```

The validation compares the headline results, common-population PEB distributions, DESIGN-selected TEST results, selected representatives, and urban geometry summaries.

The numerical comparison uses absolute and relative tolerances of `1e-8`.

## 5. Publication tables and figures

The following scripts can be executed independently using the reproduced scientific outputs:

```bash
python analysis/build_tables.py
python analysis/plot_design_tradeoffs.py
python analysis/plot_common_population.py
python analysis/plot_seed_variability.py
python analysis/plot_selected_layouts.py --icon assets/haps_icon_hq_full.png --pdf
```

The validation notebook, `notebooks/validation.ipynb`, provides additional reporting checks, including treatment of non-finite PEB values.

## 6. Verification evidence

An independent numerical reproduction was performed using Python 3.12.13 and the original scientific inputs. All five reference-result comparisons passed within the specified numerical tolerances.

The manuscript tables and figures were also regenerated successfully from the reproduced results.

The numerical run's final provenance-reporting step encountered a metadata-field error; the completed scientific results were preserved and validated, and publication reporting was completed separately. The corrected notebook has not undergone another uninterrupted full execution.

## 7. Scientific scope

The experiment evaluates geometric line-of-sight availability and positioning uncertainty bounds. PEB95 represents a geometry- and ranging-based uncertainty bound rather than measured positioning error.

The study does not explicitly model multipath propagation, diffraction, atmospheric effects, or detailed receiver tracking errors.

Further information on input origins and redistribution conditions is provided in `DATA_PROVENANCE.md`.