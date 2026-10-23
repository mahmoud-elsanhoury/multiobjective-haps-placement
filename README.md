# Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments

Reproducibility repository for the paper *Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments*, presented at ACDSA 2027.

## Cite this paper

> [!IMPORTANT]
> **Please cite our ACDSA 2027 paper if you use this repository, its implementation, datasets, or results in your research.** The DOI and official publication link will be added when available.

### BibTeX

```bibtex
@inproceedings{Asif2027HAPS,
  author    = {Hafiz M. Asif and Mahmoud Elsanhoury and Naser Tarhuni and Waseem Iqbal},
  title     = {Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments},
  booktitle = {Proceedings of the International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2027)},
  address   = {Rio de Janeiro, Brazil},
  month     = feb,
  year      = {2027},
  note      = {To appear}
}
```

### IEEE citation

H. M. Asif, M. Elsanhoury, N. Tarhuni, and W. Iqbal, "Multi-Objective HAPS Placement for Improved GNSS Positioning in Dense Urban Environments," in *Proc. Int. Conf. Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2027)*, Rio de Janeiro, Brazil, Feb. 2027, to appear.

---


## Overview

This repository provides the scientific implementation, experimental configuration, input datasets, and reference results for evaluating high-altitude platform station (HAPS) placement to improve GNSS positioning in dense urban environments.

The study considers four HAPS operating at 20 km altitude across three urban regions: Shinjuku (Tokyo), Downtown Dubai, and Frankfurt Bankenviertel.

Two placement objectives are investigated:

- **Positioning recovery (R):** The proportion of previously GNSS-unsolved receiver-time cases that become geometrically solvable with HAPS assistance.
- **Positioning quality (Q):** The median horizontal 95% positioning error bound (PEB95) among recovered cases.

The evaluation includes Greedy placement, Differential Evolution, NSGA-II, and structured HAPS layouts, compared against a fixed reference arrangement.

## Experimental configuration

The experiment uses OpenStreetMap-derived building geometry, GPS and Galileo broadcast ephemerides, and geometric line-of-sight analysis.

The evaluation contains 25 epochs on 30 August 2026, separated into 18 DESIGN and 7 held-out TEST epochs. Placement optimization and representative selection use DESIGN data exclusively. TEST data are reserved for final evaluation.

The complete scientific configuration is available in `config/experiment.json`.

## Repository contents

| Directory | Description |
|---|---|
| `notebooks/` | Main experiment and validation notebooks |
| `src/haps_gnss/` | Scientific geometry, ephemeris, and positioning modules |
| `config/` | Experimental configuration and parameters |
| `data/` | GNSS inputs, urban building geometry, and metadata |
| `run/` | Reproduction launcher and input verification |
| `analysis/` | Results validation, manuscript tables, and figure generation |
| `reference_results/` | Original results for comparison |
| `assets/` | Visualization assets |
| `docs/` | Reproducibility and data provenance documentation |

## Reproducing the experiment

Python 3.12 and the package versions in `requirements.txt` are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the scientific inputs:

```bash
python run/verify_inputs.py
```

Run the scientific smoke test:

```bash
python run/reproduce.py --smoke
```

Execute the complete experiment:

```bash
python run/reproduce.py
```

Compare the reproduced results against the reference data:

```bash
python analysis/validate_results.py
```

The full simulation is computationally intensive and is recommended for execution on a suitable HPC system.

See `docs/REPRODUCIBILITY.md` for additional execution details.

## Reproducibility verification

An independent numerical reproduction was performed using the repository's scientific implementation and original input datasets.

The reproduced headline results, common-population PEB distributions, DESIGN-selected TEST results, representative layouts, and urban geometry summaries matched the original reference results within the validation script's numerical tolerances (`atol=1e-8`, `rtol=1e-8`).

The publication tables and figures were also regenerated from the reproduced results.

## Citation

If you use this repository in academic work, please cite the associated ACDSA 2027 paper. Citation metadata are provided in `CITATION.cff`.

## Licensing



The original research software in this repository is distributed under the MIT License. See `LICENSE`.



Third-party datasets retain their respective licensing conditions:



- **OpenStreetMap building geometry:** © OpenStreetMap contributors, licensed under the Open Database License (ODbL 1.0).

- **GNSS navigation data:** Broadcast navigation data associated with the BKG/IGS data ecosystem. The precise download archive was not recorded; applicable attribution and redistribution conditions must be confirmed before public distribution.



The MIT software license does not override third-party dataset licenses.



See `docs/DATA_PROVENANCE.md` for dataset provenance, attribution, and redistribution information.
