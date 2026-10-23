#!/usr/bin/env python3
"""
Render the six paper-reproduction table-wide HAPS configurations.

Baseline: plot_bold_haps_layouts_paper_v4.py.
Scientific correction: winners are selected from DESIGN metrics only, never
from TEST outcomes. TEST metrics are attached only for evidence/reporting.

Outputs:
  selected_haps_layouts/selected_global_design_winners.csv
  selected_haps_layouts/<city>__max_R__<method>.png
  selected_haps_layouts/<city>__min_PEB__<method>.png
  optional PDFs with --pdf
"""

from __future__ import annotations

import argparse
import gc
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

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
OVERLAP_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def effective_test_peb(df: pd.DataFrame) -> pd.Series:
    strict = pd.to_numeric(df.get("test_conditional_median_PEB95_m"), errors="coerce")
    if "finite_case_median_PEB95_m" not in df.columns:
        return strict
    f = pd.to_numeric(df["finite_case_median_PEB95_m"], errors="coerce")
    return f.where(np.isfinite(f), strict)


def select_six_design_winners(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid in CITY_ORDER:
        g = reps[reps.roi_id == rid].copy()
        g["_r"] = pd.to_numeric(g.design_f1, errors="coerce")
        g["_p"] = pd.to_numeric(g.design_f2, errors="coerce")
        if g.empty:
            raise RuntimeError(f"No DESIGN representatives for {rid}")

        # Max R: DESIGN only; lower finite DESIGN PEB is deterministic tie-break.
        maxr = g["_r"].max()
        c = g[np.isclose(g["_r"], maxr, rtol=0, atol=1e-12)].copy()
        c["_ps"] = np.where(np.isfinite(c["_p"]), c["_p"], np.inf)
        r = c.sort_values(["_ps", "algorithm", "placement_id"]).iloc[0].copy()
        r["metric_kind"] = "max_R"
        rows.append(r)

        # Min PEB: DESIGN only; higher R is tie-break.
        finite = g[np.isfinite(g["_p"])].copy()
        minp = finite["_p"].min()
        c = finite[np.isclose(finite["_p"], minp, rtol=0, atol=1e-9)].copy()
        p = c.sort_values(["_r", "algorithm", "placement_id"], ascending=[False, True, True]).iloc[0].copy()
        p["metric_kind"] = "min_PEB"
        rows.append(p)

    out = pd.DataFrame(rows).reset_index(drop=True)
    if len(out) != 6:
        raise RuntimeError(f"Expected six DESIGN-selected global winners; got {len(out)}")
    return out


def prepare_icon_slot(icon_path: Path, slot_width_px: int, slot_height_px: int) -> np.ndarray:
    img = Image.open(icon_path).convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    if img.width <= 0 or img.height <= 0:
        raise RuntimeError("HAPS icon has no visible pixels")
    scale = min(slot_width_px/img.width, slot_height_px/img.height)
    nw = max(1, int(round(img.width*scale)))
    nh = max(1, int(round(img.height*scale)))
    resized = img.resize((nw,nh), Image.Resampling.LANCZOS)
    slot = Image.new("RGBA", (slot_width_px, slot_height_px), (255,255,255,0))
    slot.alpha_composite(resized, ((slot_width_px-nw)//2, (slot_height_px-nh)//2))
    return np.asarray(slot)


def compute_xy(row: pd.Series):
    theta = np.array([float(row[f"theta_{i}"]) for i in range(1,5)], float)
    radius = np.array([float(row[f"r_{i}_m"]) for i in range(1,5)], float)
    return radius*np.cos(theta), radius*np.sin(theta)


def close_groups(east: np.ndarray, north: np.ndarray, threshold_m: float):
    n = len(east)
    adj = {i:set() for i in range(n)}
    for i in range(n):
        for j in range(i+1,n):
            if math.hypot(float(east[i]-east[j]), float(north[i]-north[j])) <= threshold_m:
                adj[i].add(j); adj[j].add(i)
    groups=[]; visited=set()
    for i in range(n):
        if i in visited: continue
        stack=[i]; group=[]
        while stack:
            k=stack.pop()
            if k in visited: continue
            visited.add(k); group.append(k); stack.extend(adj[k]-visited)
        groups.append(sorted(group))
    return groups


def render_layout(row: pd.Series, icon_slot: np.ndarray, out_png: Path, out_pdf: Path|None,
                  dpi: int, figsize_in: float) -> None:
    east, north = compute_xy(row)
    radial = np.hypot(east,north)
    max_r = max(float(np.max(radial)), 1.0)
    circle_r = 1.10*max_r
    lim = 1.18*circle_r
    urban_half = 0.18*circle_r

    fig, ax = plt.subplots(figsize=(figsize_in,figsize_in), dpi=100)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_aspect("equal",adjustable="box"); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.axis("off")

    ax.add_patch(Circle((0,0), circle_r, fill=False, edgecolor="0.30", linewidth=1.35,
                        linestyle=(0,(6,4)), zorder=1))
    ax.add_patch(Rectangle((-urban_half,-urban_half), 2*urban_half,2*urban_half, fill=False,
                           edgecolor="#1559E6", linewidth=1.45, linestyle=":", zorder=2))
    for x,y in zip(east,north):
        ax.plot([0,float(x)],[0,float(y)],color="0.62",linewidth=0.85,linestyle=(0,(3,3)),zorder=1.5)

    groups = close_groups(east,north,0.08*circle_r)
    offsets={i:(0.0,0.0) for i in range(4)}; overlap_color={}
    for group in groups:
        if len(group)<=1: continue
        angles=np.linspace(0,2*math.pi,len(group),endpoint=False)
        for k,idx in enumerate(group):
            offsets[idx]=(9.0*math.cos(float(angles[k])),9.0*math.sin(float(angles[k])))
            overlap_color[idx]=OVERLAP_COLORS[k%len(OVERLAP_COLORS)]
            ax.scatter([float(east[idx])],[float(north[idx])],s=42+24*k,facecolors="none",
                       edgecolors=overlap_color[idx],linewidths=1.25,zorder=4)

    for i,(x,y) in enumerate(zip(east,north)):
        dx,dy=offsets[i]
        oi=OffsetImage(icon_slot,zoom=1.0,interpolation="lanczos",dpi_cor=False)
        arrowprops=None
        if i in overlap_color and (dx!=0 or dy!=0):
            arrowprops={"arrowstyle":"-","color":overlap_color[i],"linewidth":1.0,"shrinkA":0,"shrinkB":0}
        ab=AnnotationBbox(oi,(float(x),float(y)),xybox=(dx,dy),xycoords="data",boxcoords="offset points",
                          frameon=False,pad=0.0,arrowprops=arrowprops,zorder=10)
        ax.add_artist(ab)

    fig.subplots_adjust(left=0.015,right=0.985,bottom=0.015,top=0.985)
    fig.savefig(out_png,dpi=dpi,facecolor="white",bbox_inches="tight",pad_inches=0.01)
    if out_pdf is not None:
        fig.savefig(out_pdf,facecolor="white",bbox_inches="tight",pad_inches=0.01)
    plt.close(fig); del fig,ax; gc.collect()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+","_",str(s)).strip("_")


def main() -> None:
    ap=argparse.ArgumentParser(description="Render six DESIGN-selected paper-reproduction HAPS layouts.")
    ap.add_argument("--run-root",default="outputs/reproduction_run")
    ap.add_argument("--icon",default="haps_icon_hq_full.png")
    ap.add_argument("--outdir",default="publication_outputs/figures/selected_haps_layouts")
    ap.add_argument("--dpi",type=int,default=400)
    ap.add_argument("--slot-width",type=int,default=96)
    ap.add_argument("--slot-height",type=int,default=60)
    ap.add_argument("--figsize",type=float,default=5.0)
    ap.add_argument("--pdf",action="store_true")
    args=ap.parse_args()

    root=Path(args.run_root).expanduser().resolve(); raw=root/"raw"; audit=root/"publication_audit"
    outdir=Path(args.outdir).expanduser().resolve(); outdir.mkdir(parents=True,exist_ok=True)
    reps=read_required(raw/"paper_representatives_design_selected.csv")
    winners=select_six_design_winners(reps)

    report_path=audit/"camera_ready_reporting_table_with_peb_audit.csv"
    report=read_required(report_path if report_path.is_file() else raw/"camera_ready_headline_results.csv")
    report["test_report_PEB95_m"]=effective_test_peb(report)
    keys=["roi_id","algorithm","seed","placement_id"]
    winners=winners.merge(report,on=keys,how="left",validate="one_to_one",suffixes=("_design",""))
    winners["test_R_pct"]=100*pd.to_numeric(winners["test_R"],errors="coerce")
    winners["method"]=winners["algorithm"].map(METHOD_LABEL).fillna(winners["algorithm"])

    # Coordinates must come from the immutable DESIGN record.
    for i in range(1,5):
        src_theta=f"theta_{i}_design" if f"theta_{i}_design" in winners.columns else f"theta_{i}"
        src_r=f"r_{i}_m_design" if f"r_{i}_m_design" in winners.columns else f"r_{i}_m"
        winners[f"theta_{i}"]=pd.to_numeric(winners[src_theta],errors="coerce")
        winners[f"r_{i}_m"]=pd.to_numeric(winners[src_r],errors="coerce")
    if winners[[f"theta_{i}" for i in range(1,5)]+[f"r_{i}_m" for i in range(1,5)]].isna().any().any():
        raise RuntimeError("A selected winner is missing immutable DESIGN coordinates")

    for idx,row in winners.iterrows():
        e,n=compute_xy(row)
        for i in range(4):
            winners.loc[idx,f"east_{i+1}_m"]=float(e[i]); winners.loc[idx,f"north_{i+1}_m"]=float(n[i])

    winners.to_csv(outdir/"selected_global_design_winners.csv",index=False)

    icon=Path(args.icon).expanduser().resolve()
    if not icon.is_file():
        raise FileNotFoundError(f"HAPS icon not found: {icon}")
    slot=prepare_icon_slot(icon,args.slot_width,args.slot_height)

    print("Selected global DESIGN winners:")
    for _,row in winners.iterrows():
        if row.metric_kind=="max_R":
            metric=f"DESIGN R={100*float(row.design_f1):.4f}% | TEST R={float(row.test_R_pct):.4f}%"
        else:
            metric=f"DESIGN PEB={float(row.design_f2):.4f} m | TEST PEB={float(row.test_report_PEB95_m):.4f} m"
        print(f"  {CITY_LABEL[row.roi_id]:<10} {row.metric_kind:<7} {row.method:<17} {metric}")

    for _,row in winners.iterrows():
        stem=f"{row.roi_id}__{row.metric_kind}__{safe_name(row.method)}"
        png=outdir/f"{stem}.png"; pdf=outdir/f"{stem}.pdf" if args.pdf else None
        print(f"Rendering {png.name} ...",flush=True)
        render_layout(row,slot,png,pdf,args.dpi,args.figsize)

    print(f"[OK] Generated exactly {len(winners)} DESIGN-selected winner figures")
    print(f"Evidence CSV: {outdir/'selected_global_design_winners.csv'}")


if __name__=="__main__":
    main()
