#!/usr/bin/env python3
"""
Plot TEST variability across DESIGN-fixed stochastic representatives.

Input:
  raw/seed_test_statistics.csv

The file is already based on per-seed representatives selected from DESIGN.
This script performs no seed selection from TEST outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CITY_ORDER = ["shinjuku_tokyo", "downtown_dubai", "frankfurt_bankenviertel"]
CITY_LABEL = {
    "shinjuku_tokyo": "Shinjuku",
    "downtown_dubai": "Dubai",
    "frankfurt_bankenviertel": "Frankfurt",
}
METHOD_LABEL = {
    "differential_evolution": "DE",
    "nsga2": "NSGA-II",
}
RULE_LABEL = {
    "design_R_solution": "R solution",
    "design_max_R": "max R",
    "design_min_PEB": "min PEB",
}


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def series_label(row: pd.Series) -> str:
    method = METHOD_LABEL.get(row.algorithm, row.algorithm)
    rule = RULE_LABEL.get(row.selection_rule, row.selection_rule)
    return f"{method} ({rule})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot stochastic seed variability.")
    ap.add_argument("--run-root", default="outputs/reproduction_run")
    ap.add_argument("--outdir", default="publication_outputs/figures")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    raw = root / "raw"
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_required(raw / "seed_test_statistics.csv")
    needed = {
        "roi_id", "algorithm", "selection_rule", "n_seeds",
        "test_R_mean", "test_R_std", "test_PEB_mean", "test_PEB_std",
    }
    if not needed.issubset(d.columns):
        raise RuntimeError(f"seed_test_statistics.csv missing: {sorted(needed-set(d.columns))}")

    d = d[d.roi_id.isin(CITY_ORDER)].copy()
    d["test_R_mean_pct"] = 100*pd.to_numeric(d.test_R_mean, errors="coerce")
    d["test_R_std_pct"] = 100*pd.to_numeric(d.test_R_std, errors="coerce")
    d["test_PEB_mean"] = pd.to_numeric(d.test_PEB_mean, errors="coerce")
    d["test_PEB_std"] = pd.to_numeric(d.test_PEB_std, errors="coerce")
    d["label"] = d.apply(series_label, axis=1)
    d.to_csv(outdir / "seed_variability_plot_data.csv", index=False)

    # Keep a fixed category order across cities.
    categories = []
    preferred = [
        ("differential_evolution", "design_R_solution"),
        ("nsga2", "design_max_R"),
        ("nsga2", "design_min_PEB"),
    ]
    for algo, rule in preferred:
        label = f"{METHOD_LABEL.get(algo,algo)} ({RULE_LABEL.get(rule,rule)})"
        if label in set(d.label):
            categories.append(label)
    for label in d.label.drop_duplicates():
        if label not in categories:
            categories.append(label)

    # Figure 1: R mean ± std.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=False)
    for ax, rid in zip(axes, CITY_ORDER):
        g = d[d.roi_id == rid].set_index("label")
        vals=[]; errs=[]; labs=[]
        for lab in categories:
            if lab not in g.index:
                continue
            r = g.loc[lab]
            if isinstance(r, pd.DataFrame): r=r.iloc[0]
            vals.append(float(r.test_R_mean_pct)); errs.append(float(r.test_R_std_pct)); labs.append(lab)
        x=np.arange(len(labs))
        ax.bar(x, vals, yerr=errs, capsize=4, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
        ax.set_title(CITY_LABEL[rid]); ax.set_ylabel("TEST R mean ± std [%]")
        ax.grid(axis="y",alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir/"seed_variability_R.png",dpi=args.dpi,bbox_inches="tight")
    fig.savefig(outdir/"seed_variability_R.pdf",dpi=args.dpi,bbox_inches="tight")
    plt.close(fig)

    # Figure 2: conditional PEB mean ± std. Rows with unavailable aggregate PEB are skipped.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=False)
    for ax, rid in zip(axes, CITY_ORDER):
        g = d[d.roi_id == rid].set_index("label")
        vals=[]; errs=[]; labs=[]
        for lab in categories:
            if lab not in g.index:
                continue
            r = g.loc[lab]
            if isinstance(r, pd.DataFrame): r=r.iloc[0]
            v=float(r.test_PEB_mean) if np.isfinite(r.test_PEB_mean) else np.nan
            e=float(r.test_PEB_std) if np.isfinite(r.test_PEB_std) else np.nan
            if not np.isfinite(v):
                continue
            vals.append(v); errs.append(e if np.isfinite(e) else 0.0); labs.append(lab)
        x=np.arange(len(labs))
        ax.bar(x, vals, yerr=errs, capsize=4, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
        ax.set_title(CITY_LABEL[rid]); ax.set_ylabel(r"TEST conditional median PEB$_{95,h}$ mean ± std [m]")
        ax.grid(axis="y",alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir/"seed_variability_PEB.png",dpi=args.dpi,bbox_inches="tight")
    fig.savefig(outdir/"seed_variability_PEB.pdf",dpi=args.dpi,bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Seed variability figures saved under {outdir}")


if __name__ == "__main__":
    main()
