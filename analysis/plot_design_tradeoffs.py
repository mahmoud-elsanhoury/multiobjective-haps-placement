#!/usr/bin/env python3
"""
Plot DESIGN-space HAPS trade-offs for the three paper-reproduction cities.

Unlike the old held-out/test Pareto script, this version never selects or
highlights layouts from TEST outcomes. It uses:
  raw/frozen_design_layouts.csv
  raw/paper_representatives_design_selected.csv

The figure keeps the old paper theme: one row per city, trade-off scatter on
the left and miniatures of the two global DESIGN winners on the right.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, Rectangle

CITY_ORDER = ["shinjuku_tokyo", "downtown_dubai", "frankfurt_bankenviertel"]
CITY_LABEL = {
    "shinjuku_tokyo": "Shinjuku",
    "downtown_dubai": "Dubai",
    "frankfurt_bankenviertel": "Frankfurt",
}
METHOD_LABEL = {
    "greedy_add": "Greedy",
    "differential_evolution": "DE",
    "nsga2": "NSGA-II",
    "symmetric_ring": "Symmetric ring",
    "center_plus_three": "Center + three",
}
METHOD_ORDER = ["greedy_add", "differential_evolution", "nsga2", "symmetric_ring", "center_plus_three"]
METHOD_MARKERS = {
    "greedy_add": "s",
    "differential_evolution": "^",
    "nsga2": "o",
    "symmetric_ring": "D",
    "center_plus_three": "P",
}
METHOD_COLORS = {
    "greedy_add": "0.35",
    "differential_evolution": "tab:orange",
    "nsga2": "tab:blue",
    "symmetric_ring": "tab:green",
    "center_plus_three": "tab:red",
}


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def select_global_design_winners(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid in CITY_ORDER:
        g = reps[reps.roi_id == rid].copy()
        if g.empty:
            raise RuntimeError(f"No representatives for {rid}")
        g["_r"] = pd.to_numeric(g.design_f1, errors="coerce")
        g["_p"] = pd.to_numeric(g.design_f2, errors="coerce")

        maxr = g["_r"].max()
        c = g[np.isclose(g["_r"], maxr, rtol=0, atol=1e-12)].copy()
        c["_ps"] = np.where(np.isfinite(c["_p"]), c["_p"], np.inf)
        r = c.sort_values(["_ps", "algorithm", "placement_id"]).iloc[0].copy()
        r["metric_kind"] = "max_R"
        rows.append(r)

        finite = g[np.isfinite(g["_p"])].copy()
        minp = finite["_p"].min()
        c = finite[np.isclose(finite["_p"], minp, rtol=0, atol=1e-9)].copy()
        p = c.sort_values(["_r", "algorithm", "placement_id"], ascending=[False, True, True]).iloc[0].copy()
        p["metric_kind"] = "min_PEB"
        rows.append(p)
    return pd.DataFrame(rows).reset_index(drop=True)


def nondominated(g: pd.DataFrame) -> pd.DataFrame:
    d = g.copy().reset_index(drop=True)
    d["design_f1"] = pd.to_numeric(d.design_f1, errors="coerce")
    d["design_f2"] = pd.to_numeric(d.design_f2, errors="coerce")
    d = d[np.isfinite(d.design_f1) & np.isfinite(d.design_f2)].copy().reset_index(drop=True)
    if d.empty:
        return d
    f1 = d.design_f1.to_numpy(float)
    f2 = d.design_f2.to_numpy(float)
    keep = np.ones(len(d), bool)
    for i in range(len(d)):
        dom = (f1 >= f1[i]) & (f2 <= f2[i]) & ((f1 > f1[i]) | (f2 < f2[i]))
        dom[i] = False
        if dom.any():
            keep[i] = False
    return d.loc[keep].copy()


def draw_layout_miniature(ax, row: pd.Series) -> None:
    theta = np.array([row[f"theta_{i}"] for i in range(1,5)], float)
    radius = np.array([row[f"r_{i}_m"] for i in range(1,5)], float)
    east = radius * np.cos(theta)
    north = radius * np.sin(theta)
    maxr = max(float(np.nanmax(radius)), 1.0)
    lim = 1.18 * maxr

    ax.add_patch(Circle((0,0), 1.08*maxr, fill=False, lw=1.0, ls="--", color="0.45"))
    sq = 0.22 * maxr
    ax.add_patch(Rectangle((-sq,-sq), 2*sq, 2*sq, fill=False, lw=1.0, ls=":", edgecolor="tab:blue"))
    for e,n in zip(east,north):
        ax.plot([0,e],[0,n], ls="--", lw=0.8, color="0.65", zorder=1)
    ax.scatter(east,north,s=34,marker="o",facecolor=METHOD_COLORS.get(row.algorithm,"tab:blue"),edgecolor="black",linewidth=0.6,zorder=3)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect("equal",adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    title = METHOD_LABEL.get(row.algorithm,row.algorithm)
    if row.metric_kind == "max_R":
        metric = f"DESIGN max R = {100*row.design_f1:.2f}%"
    else:
        metric = f"DESIGN min PEB = {row.design_f2:.1f} m"
    ax.set_title(f"{title}\n{metric}", fontsize=8, pad=3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot paper-reproduction DESIGN Pareto/trade-off figure.")
    ap.add_argument("--run-root", default="outputs/reproduction_run")
    ap.add_argument("--outdir", default="publication_outputs/figures")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    raw = root / "raw"
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    frozen = read_required(raw / "frozen_design_layouts.csv")
    reps = read_required(raw / "paper_representatives_design_selected.csv")
    winners = select_global_design_winners(reps)
    winners.to_csv(outdir / "design_pareto_highlighted_winners.csv", index=False)

    fig = plt.figure(figsize=(12.8, 12.3))
    outer = gridspec.GridSpec(3, 2, width_ratios=[3.5, 2.2], hspace=0.34, wspace=0.16)

    for i, rid in enumerate(CITY_ORDER):
        ax = fig.add_subplot(outer[i,0])
        city = frozen[frozen.roi_id == rid].copy()
        city["design_f1"] = pd.to_numeric(city.design_f1, errors="coerce")
        city["design_f2"] = pd.to_numeric(city.design_f2, errors="coerce")

        for algo in METHOD_ORDER:
            g = city[city.algorithm == algo]
            finite = g[np.isfinite(g.design_f1) & np.isfinite(g.design_f2)]
            if finite.empty:
                continue
            ax.scatter(
                finite.design_f2, 100*finite.design_f1,
                s=25, marker=METHOD_MARKERS[algo], c=METHOD_COLORS[algo],
                alpha=0.62, edgecolors="none", label=METHOD_LABEL[algo], zorder=2,
            )
            # Connect only the non-dominated subset for genuinely multi-objective families.
            if algo in {"nsga2","symmetric_ring","center_plus_three"}:
                nd = nondominated(finite).sort_values("design_f2")
                if len(nd) >= 2:
                    ax.plot(nd.design_f2, 100*nd.design_f1, lw=0.8, alpha=0.45, color=METHOD_COLORS[algo], zorder=1)

        hi = winners[winners.roi_id == rid]
        for _, r in hi.iterrows():
            if np.isfinite(float(r.design_f2)):
                ax.scatter(
                    [float(r.design_f2)], [100*float(r.design_f1)],
                    s=165, marker=METHOD_MARKERS.get(r.algorithm,"o"),
                    facecolors="none", edgecolors="black", linewidths=1.7, zorder=5,
                )
                label = "DESIGN max R" if r.metric_kind == "max_R" else "DESIGN min PEB"
                ax.annotate(label, (float(r.design_f2), 100*float(r.design_f1)), xytext=(5,5), textcoords="offset points", fontsize=7)

        omitted = city[np.isfinite(city.design_f1) & ~np.isfinite(city.design_f2)]
        if len(omitted):
            ax.text(0.99, 0.02, f"{len(omitted)} finite-R layout(s) omitted: DESIGN PEB undefined",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.35")

        ax.set_title(CITY_LABEL[rid], fontsize=11)
        ax.set_xlabel(r"DESIGN median PEB$_{95,h}$ [m]")
        ax.set_ylabel(r"DESIGN solvable fraction $R$ [%]")
        ax.grid(True, alpha=0.25, lw=0.5)
        if i == 0:
            ax.legend(fontsize=8, loc="best", frameon=True)

        inner = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[i,1],hspace=0.45)
        for j, (_, r) in enumerate(hi.sort_values("metric_kind", ascending=False).iterrows()):
            axm = fig.add_subplot(inner[j,0])
            draw_layout_miniature(axm, r)

    fig.suptitle("DESIGN-space HAPS placement trade-offs and immutable selected winners", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.96, hspace=0.34, wspace=0.16)
    pdf = outdir / "design_pareto_with_layouts.pdf"
    png = outdir / "design_pareto_with_layouts.png"
    fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {pdf}")
    print(f"[OK] Saved {png}")


if __name__ == "__main__":
    main()
