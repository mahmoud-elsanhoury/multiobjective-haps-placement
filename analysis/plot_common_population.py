#!/usr/bin/env python3
"""
Plot the fixed-denominator common-population TEST quality metric.

For threshold tau, the curve is the fraction of ALL GNSS-unsolved TEST cases
that are solved by the selected layout with PEB95,h <= tau. Non-recovered and
non-finite cases remain outside the numerator, so every method shares the same
denominator.

Only immutable DESIGN-selected paper representatives plus the preserved
reference are plotted. TEST data are never used for selection.
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
    "preserved_c4_reference": "Reference",
    "greedy_add": "Greedy",
    "differential_evolution": "DE",
    "nsga2": "NSGA-II",
    "symmetric_ring": "Symmetric ring",
    "center_plus_three": "Center + three",
}
RULE_LABEL = {
    "reference": "reference",
    "design_R_solution": "R solution",
    "design_max_R": "max R",
    "design_min_PEB": "min PEB",
}
LINESTYLE = {
    "reference": "--",
    "design_R_solution": "-",
    "design_max_R": "-",
    "design_min_PEB": ":",
}


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot common-population PEB quality curves.")
    ap.add_argument("--run-root", default="outputs/reproduction_run")
    ap.add_argument("--outdir", default="publication_outputs/figures")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    raw = root / "raw"
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ecdf = read_required(raw / "common_population_peb_ecdf.csv")
    reps = read_required(raw / "paper_representatives_design_selected.csv")
    ref = read_required(raw / "fixed_reference_results.csv")

    needed = {"peb95_threshold_m", "fraction_all_gnss_unsolved", "roi_id", "algorithm", "seed", "placement_id"}
    if not needed.issubset(ecdf.columns):
        raise RuntimeError(f"common_population_peb_ecdf.csv missing: {sorted(needed-set(ecdf.columns))}")

    # Build the exact allowed immutable paper set.
    allowed = reps[["roi_id","algorithm","seed","placement_id","selection_rule"]].copy()
    rref = ref[["roi_id","algorithm","seed","placement_id"]].copy()
    rref["selection_rule"] = "reference"
    allowed = pd.concat([allowed, rref], ignore_index=True)

    # If one physical layout is simultaneously DESIGN max-R and min-PEB, keep
    # one curve and preserve both reasons in its label.
    key_cols = ["roi_id","algorithm","seed","placement_id"]
    allowed = (allowed.groupby(key_cols, dropna=False)["selection_rule"]
               .agg(lambda x: "+".join(dict.fromkeys(map(str, x))))
               .reset_index())

    curves = ecdf.merge(
        allowed,
        on=key_cols,
        how="inner",
        validate="many_to_one",
    )
    if curves.empty:
        raise RuntimeError("No common-population curves matched immutable paper representatives")

    curves["peb95_threshold_m"] = pd.to_numeric(curves["peb95_threshold_m"], errors="coerce")
    curves["fraction_all_gnss_unsolved"] = pd.to_numeric(curves["fraction_all_gnss_unsolved"], errors="coerce")
    curves = curves[np.isfinite(curves.peb95_threshold_m) & np.isfinite(curves.fraction_all_gnss_unsolved)].copy()
    curves.to_csv(outdir / "common_population_quality_paper_curves.csv", index=False)

    # Combined 3-panel paper figure.
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for ax, rid in zip(axes, CITY_ORDER):
        city = curves[curves.roi_id == rid]
        for (algo, rule, pid), g in city.groupby(["algorithm","selection_rule","placement_id"], sort=False):
            g = g.sort_values("peb95_threshold_m")
            label = METHOD_LABEL.get(algo, algo)
            if rule != "reference":
                label += f" ({RULE_LABEL.get(rule,rule)})"
            ax.step(
                g.peb95_threshold_m,
                100*g.fraction_all_gnss_unsolved,
                where="post",
                linewidth=1.45,
                linestyle=LINESTYLE.get(rule, "-." if "+" in str(rule) else "-"),
                label=label,
                alpha=0.9,
            )
        ax.set_xscale("log")
        ax.set_title(CITY_LABEL[rid])
        ax.set_xlabel(r"PEB$_{95,h}$ threshold [m]")
        ax.grid(True, alpha=0.25, which="both")
    axes[0].set_ylabel("All GNSS-unsolved TEST cases solved\nwithin threshold [%]")
    axes[-1].legend(fontsize=6.8, loc="best", frameon=True)
    fig.tight_layout()
    pdf = outdir / "common_population_quality.pdf"
    png = outdir / "common_population_quality.png"
    fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Per-city versions are useful when the combined legend is too dense.
    for rid in CITY_ORDER:
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        city = curves[curves.roi_id == rid]
        for (algo, rule, pid), g in city.groupby(["algorithm","selection_rule","placement_id"], sort=False):
            g = g.sort_values("peb95_threshold_m")
            label = METHOD_LABEL.get(algo, algo)
            if rule != "reference":
                label += f" ({RULE_LABEL.get(rule,rule)})"
            ax.step(g.peb95_threshold_m,100*g.fraction_all_gnss_unsolved,where="post",
                    linewidth=1.5,linestyle=LINESTYLE.get(rule, "-." if "+" in str(rule) else "-"),label=label,alpha=0.9)
        ax.set_xscale("log")
        ax.set_title(CITY_LABEL[rid])
        ax.set_xlabel(r"PEB$_{95,h}$ threshold [m]")
        ax.set_ylabel("All GNSS-unsolved TEST cases solved within threshold [%]")
        ax.grid(True,alpha=0.25,which="both")
        ax.legend(fontsize=7,loc="best",frameon=True)
        fig.tight_layout()
        fig.savefig(outdir/f"common_population_quality_{rid}.png",dpi=args.dpi,bbox_inches="tight")
        fig.savefig(outdir/f"common_population_quality_{rid}.pdf",dpi=args.dpi,bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Common-population figures saved under {outdir}")


if __name__ == "__main__":
    main()
