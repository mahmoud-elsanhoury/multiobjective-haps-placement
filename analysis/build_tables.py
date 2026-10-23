#!/usr/bin/env python3
"""
Build paper-reproduction result tables from the completed ACDSA 2027 V2 experiment.

Scientific rule
---------------
Layout/seed selection is NEVER performed from TEST metrics. The script uses the
immutable DESIGN-selected representatives and only attaches their TEST outcomes.

Primary inputs (under --run-root):
  raw/paper_representatives_design_selected.csv
  raw/fixed_reference_results.csv
  publication_audit/camera_ready_reporting_table_with_peb_audit.csv
  raw/seed_test_statistics.csv

Outputs (under --outdir):
  table_main_results_long.csv
  table_main_results_wide.csv
  table_global_winners_design_selected.csv
  table_improvements_vs_reference.csv
  table_seed_variability.csv
  table_main_results.tex
  TABLE_BUILD_SUMMARY.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CITY_ORDER = [
    "shinjuku_tokyo",
    "downtown_dubai",
    "frankfurt_bankenviertel",
]
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
METHOD_ORDER = {
    "preserved_c4_reference": 0,
    "greedy_add": 1,
    "differential_evolution": 2,
    "nsga2": 3,
    "symmetric_ring": 4,
    "center_plus_three": 5,
}
SELECTION_LABEL = {
    "reference": "Reference",
    "design_R_solution": "R solution",
    "design_max_R": "max R",
    "design_min_PEB": "min PEB",
}


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def effective_test_peb(df: pd.DataFrame) -> pd.Series:
    """Publication reporting PEB: finite-case median if audit provides it.

    This is descriptive reporting only. It does not alter DESIGN optimization.
    """
    strict = pd.to_numeric(df.get("test_conditional_median_PEB95_m"), errors="coerce")
    if "finite_case_median_PEB95_m" not in df.columns:
        return strict
    finite_case = pd.to_numeric(df["finite_case_median_PEB95_m"], errors="coerce")
    return finite_case.where(np.isfinite(finite_case), strict)


def select_global_design_winners(paper_reps: pd.DataFrame) -> pd.DataFrame:
    """Choose exactly two table-wide winners per city using DESIGN metrics only.

    Reference is excluded, matching the old six-layout figure logic.
    """
    rows = []
    for rid in CITY_ORDER:
        g = paper_reps[paper_reps["roi_id"] == rid].copy()
        if g.empty:
            raise RuntimeError(f"No DESIGN-selected representatives found for {rid}")

        # DESIGN max-R. Deterministic tie-break: finite/lower DESIGN PEB, then names.
        g["_d1"] = pd.to_numeric(g["design_f1"], errors="coerce")
        g["_d2"] = pd.to_numeric(g["design_f2"], errors="coerce")
        valid_r = g[np.isfinite(g["_d1"])].copy()
        if valid_r.empty:
            raise RuntimeError(f"No finite DESIGN R for {rid}")
        max_r = valid_r["_d1"].max()
        cand = valid_r[np.isclose(valid_r["_d1"], max_r, rtol=0.0, atol=1e-12)].copy()
        cand["_d2_sort"] = np.where(np.isfinite(cand["_d2"]), cand["_d2"], np.inf)
        win_r = cand.sort_values(
            ["_d2_sort", "algorithm", "placement_id"],
            ascending=[True, True, True],
        ).iloc[0].copy()
        win_r["global_metric_kind"] = "max_R"
        rows.append(win_r)

        # DESIGN min-PEB. Only finite DESIGN PEB can compete.
        valid_p = g[np.isfinite(g["_d2"])].copy()
        if valid_p.empty:
            raise RuntimeError(f"No finite DESIGN PEB for {rid}")
        min_p = valid_p["_d2"].min()
        cand = valid_p[np.isclose(valid_p["_d2"], min_p, rtol=0.0, atol=1e-9)].copy()
        win_p = cand.sort_values(
            ["_d1", "algorithm", "placement_id"],
            ascending=[False, True, True],
        ).iloc[0].copy()
        win_p["global_metric_kind"] = "min_PEB"
        rows.append(win_p)

    out = pd.DataFrame(rows).drop(columns=[c for c in ["_d1", "_d2", "_d2_sort"] if c in pd.DataFrame(rows).columns], errors="ignore")
    if len(out) != 6:
        raise RuntimeError(f"Expected exactly six global DESIGN winners, got {len(out)}")
    return out.reset_index(drop=True)


def latex_escape(s: str) -> str:
    return (str(s)
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_"))


def build_latex(main_long: pd.DataFrame, out_path: Path) -> None:
    """Write a compact IEEE-friendly LaTeX table body."""
    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"\centering")
    rows.append(r"\caption{TEST performance of immutable DESIGN-selected HAPS representatives. $R$ is shown in percent and PEB$_{95,h}$ in metres.}")
    rows.append(r"\label{tab:camera_ready_results}")
    rows.append(r"\begin{tabular}{llrrrrrr}")
    rows.append(r"\hline")
    rows.append(r"Method & Representative & \multicolumn{2}{c}{Shinjuku} & \multicolumn{2}{c}{Dubai} & \multicolumn{2}{c}{Frankfurt} \\")
    rows.append(r" & & $R$ (\%) & PEB & $R$ (\%) & PEB & $R$ (\%) & PEB \\")
    rows.append(r"\hline")

    keyed = main_long.set_index(["algorithm", "selection_rule", "roi_id"])
    row_defs = [
        ("preserved_c4_reference", "reference"),
        ("greedy_add", "design_R_solution"),
        ("differential_evolution", "design_max_R"),
        ("nsga2", "design_max_R"),
        ("nsga2", "design_min_PEB"),
        ("symmetric_ring", "design_max_R"),
        ("symmetric_ring", "design_min_PEB"),
        ("center_plus_three", "design_max_R"),
        ("center_plus_three", "design_min_PEB"),
    ]
    for algo, rule in row_defs:
        vals = []
        for rid in CITY_ORDER:
            if (algo, rule, rid) not in keyed.index:
                vals.extend(["--", "--"])
                continue
            r = keyed.loc[(algo, rule, rid)]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            vals.extend([
                f"{float(r['test_R_pct']):.2f}" if np.isfinite(r["test_R_pct"]) else "--",
                f"{float(r['test_report_PEB95_m']):.1f}" if np.isfinite(r["test_report_PEB95_m"]) else "--",
            ])
        rows.append(
            f"{latex_escape(METHOD_LABEL[algo])} & {latex_escape(SELECTION_LABEL.get(rule, rule))} & "
            + " & ".join(vals) + r" \\"
        )
    rows.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build leakage-safe paper-reproduction result tables.")
    ap.add_argument(
        "--run-root",
        default="outputs/reproduction_run",
        help="Completed paper-reproduction run directory.",
    )
    ap.add_argument(
        "--outdir",
        default="publication_outputs/tables",
        help="Output directory.",
    )
    args = ap.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    raw = root / "raw"
    audit = root / "publication_audit"
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    reps = read_required(raw / "paper_representatives_design_selected.csv")
    ref = read_required(raw / "fixed_reference_results.csv")
    seed_stats = read_required(raw / "seed_test_statistics.csv")

    audit_table_path = audit / "camera_ready_reporting_table_with_peb_audit.csv"
    if audit_table_path.is_file():
        report = pd.read_csv(audit_table_path)
        report_source = str(audit_table_path)
    else:
        report = read_required(raw / "camera_ready_headline_results.csv")
        report_source = str(raw / "camera_ready_headline_results.csv")

    # Core assertions.
    needed = {"roi_id", "algorithm", "placement_id", "design_f1", "design_f2"}
    if not needed.issubset(reps.columns):
        raise RuntimeError(f"paper representatives missing columns: {sorted(needed-set(reps.columns))}")
    if "selection_rule" not in reps.columns:
        raise RuntimeError("paper representatives must contain selection_rule")

    # Every paper representative must be represented in the reporting table.
    join_keys = ["roi_id", "algorithm", "seed", "placement_id"]
    missing_keys = reps[join_keys].merge(
        report[join_keys].drop_duplicates(), on=join_keys, how="left", indicator=True
    )
    if (missing_keys["_merge"] != "both").any():
        raise RuntimeError("At least one immutable DESIGN representative is missing from TEST reporting data")

    report = report.copy()
    report["test_R"] = pd.to_numeric(report["test_R"], errors="coerce")
    report["test_R_pct"] = 100.0 * report["test_R"]
    report["test_report_PEB95_m"] = effective_test_peb(report)
    if "nonfinite_PEB_pct" not in report.columns:
        report["nonfinite_PEB_pct"] = np.nan

    # Normalize metadata.
    report["city"] = report["roi_id"].map(CITY_LABEL).fillna(report["roi_id"])
    report["method"] = report["algorithm"].map(METHOD_LABEL).fillna(report["algorithm"])
    report["representative"] = report["selection_rule"].map(SELECTION_LABEL).fillna(report["selection_rule"])
    report["method_order"] = report["algorithm"].map(METHOD_ORDER).fillna(99)
    report["city_order"] = report["roi_id"].apply(lambda x: CITY_ORDER.index(x) if x in CITY_ORDER else 99)

    main_long = report[report["roi_id"].isin(CITY_ORDER)].copy()
    main_long = main_long.sort_values(["method_order", "selection_rule", "city_order"]).reset_index(drop=True)

    long_cols = [
        "roi_id", "city", "algorithm", "method", "seed", "placement_id", "selection_rule", "representative",
        "design_f1", "design_f2", "test_R", "test_R_pct", "test_report_PEB95_m",
        "test_conditional_median_PEB95_m", "nonfinite_PEB_pct", "n_gnss_unsolved_test", "n_made_solvable_test",
    ]
    long_cols = [c for c in long_cols if c in main_long.columns]
    main_long[long_cols].to_csv(outdir / "table_main_results_long.csv", index=False)

    # Wide manuscript-style table.
    wide_rows = []
    for (algo, rule), g in main_long.groupby(["algorithm", "selection_rule"], sort=False):
        row = {
            "algorithm": algo,
            "method": METHOD_LABEL.get(algo, algo),
            "selection_rule": rule,
            "representative": SELECTION_LABEL.get(rule, rule),
            "method_order": METHOD_ORDER.get(algo, 99),
        }
        for rid in CITY_ORDER:
            s = g[g["roi_id"] == rid]
            if len(s):
                r = s.iloc[0]
                lab = CITY_LABEL[rid].lower()
                row[f"{lab}_R_pct"] = r["test_R_pct"]
                row[f"{lab}_PEB95_m"] = r["test_report_PEB95_m"]
                row[f"{lab}_nonfinite_PEB_pct"] = r.get("nonfinite_PEB_pct", np.nan)
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows).sort_values(["method_order", "selection_rule"]).drop(columns="method_order")
    wide.to_csv(outdir / "table_main_results_wide.csv", index=False)

    # Exactly six global winners: DESIGN selection only.
    global_design = select_global_design_winners(reps)
    winner_report = global_design.merge(
        report,
        on=["roi_id", "algorithm", "seed", "placement_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_design", ""),
    )
    winner_report["test_report_PEB95_m"] = effective_test_peb(winner_report)
    winner_report["test_R_pct"] = 100.0 * pd.to_numeric(winner_report["test_R"], errors="coerce")
    winner_cols = [
        "roi_id", "algorithm", "seed", "placement_id", "global_metric_kind",
        "design_f1", "design_f2", "test_R", "test_R_pct", "test_report_PEB95_m", "nonfinite_PEB_pct",
    ] + [f"theta_{i}" for i in range(1,5)] + [f"r_{i}_m" for i in range(1,5)]
    winner_cols = [c for c in winner_cols if c in winner_report.columns]
    winner_report[winner_cols].to_csv(outdir / "table_global_winners_design_selected.csv", index=False)

    # Reference comparisons using TEST outcomes of DESIGN-selected global winners.
    improvements = []
    for rid in CITY_ORDER:
        rref = report[(report["roi_id"] == rid) & (report["algorithm"] == "preserved_c4_reference")]
        if rref.empty:
            raise RuntimeError(f"Reference result missing for {rid}")
        rref = rref.iloc[0]

        wr = winner_report[(winner_report["roi_id"] == rid) & (winner_report["global_metric_kind"] == "max_R")].iloc[0]
        wp = winner_report[(winner_report["roi_id"] == rid) & (winner_report["global_metric_kind"] == "min_PEB")].iloc[0]

        ref_R = float(rref["test_R"])
        ref_P = float(rref["test_report_PEB95_m"])
        opt_R = float(wr["test_R"])
        opt_P = float(wp["test_report_PEB95_m"])
        improvements.append({
            "roi_id": rid,
            "city": CITY_LABEL[rid],
            "reference_test_R_pct": 100.0 * ref_R,
            "design_selected_max_R_method": METHOD_LABEL.get(wr["algorithm"], wr["algorithm"]),
            "design_selected_max_R_test_R_pct": 100.0 * opt_R,
            "delta_R_percentage_points": 100.0 * (opt_R - ref_R),
            "reference_test_PEB95_m": ref_P,
            "design_selected_min_PEB_method": METHOD_LABEL.get(wp["algorithm"], wp["algorithm"]),
            "design_selected_min_PEB_test_PEB95_m": opt_P,
            "PEB_reduction_vs_reference_pct": 100.0 * (ref_P - opt_P) / ref_P if np.isfinite(ref_P) and ref_P > 0 else np.nan,
        })
    improvement_df = pd.DataFrame(improvements)
    improvement_df.to_csv(outdir / "table_improvements_vs_reference.csv", index=False)

    seed_stats.to_csv(outdir / "table_seed_variability.csv", index=False)
    build_latex(main_long, outdir / "table_main_results.tex")

    summary = {
        "run_root": str(root),
        "report_source": report_source,
        "selection_basis": "DESIGN only",
        "test_used_for_selection": False,
        "n_main_rows": int(len(main_long)),
        "n_global_winners": int(len(winner_report)),
        "files": [
            "table_main_results_long.csv",
            "table_main_results_wide.csv",
            "table_global_winners_design_selected.csv",
            "table_improvements_vs_reference.csv",
            "table_seed_variability.csv",
            "table_main_results.tex",
        ],
    }
    (outdir / "TABLE_BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] Paper-reproduction tables built")
    print(improvement_df.to_string(index=False))
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
