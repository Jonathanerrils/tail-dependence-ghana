#!/usr/bin/env python
"""
Build the publication-figure candidate suite from frozen repository outputs.

Run from the repository root:
    python scripts/37_build_publication_figures.py

Outputs:
    outputs/publication_figures/
        figure_01_calendar_workflow.{pdf,png}
        figure_02_tail_concentration_profiles.{pdf,png}
        figure_03_copula_gof_heatmap.{pdf,png}
        figure_04_stress_summary.{pdf,png}
        figure_S1_cedi_evt_threshold_sensitivity.{pdf,png}
        figure_S2_cedi_AG_sensitivity.{pdf,png}
        figure_S3_pit_clipping_diagnostic.{pdf,png}
        graphical_abstract_verified_findings.{pdf,png}
        repository_study_design_workflow.{pdf,png}
        table_01_cross_panel_evidence.{csv,md}
        figure_manifest.csv
        figure_input_hashes.txt
        figure_captions.md

This script is plotting-only. It does not refit margins, copulas, EVT models,
or bootstrap tests.

Publication-note: top-level "Figure X..." headers are deliberately not drawn
inside the image files. Figure numbering/titles belong in the LaTeX captions.
"""

from __future__ import annotations

import hashlib
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIGURES = ROOT / "paper" / "latex" / "rebuild" / "figures"
MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

DPI = 300
Q_LEVELS = [0.025, 0.05, 0.10]
PAIRS_ORDER = [
    "Cocoa-Gold",
    "Cocoa-Brent",
    "Cocoa-WTI",
    "Cocoa-Cedi",
    "Gold-Brent",
    "Gold-WTI",
    "Gold-Cedi",
    "Brent-WTI",
    "Brent-Cedi",
    "WTI-Cedi",
]


# Frozen publication inputs
PATHS = {
    "pit_A": ROOT / "outputs" / "tables" / "_pit_real.csv",
    "pit_B": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "_pit_publication_v2.csv",
    "gof_A": ROOT / "outputs" / "tables" / "copula_gof_8family_B2000_B5000.csv",
    "gof_B": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "gof_publication_v2_final.csv",
    "stress_A_combined": ROOT / "outputs" / "tables" / "calm_stress_full60_B15000.csv",
    "stress_A_disagg": ROOT / "outputs" / "tables" / "calm_stress_disaggregated_B15000.csv",
    "stress_B_combined": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "stress_combined_B15000_final_v2.csv",
    "stress_B_covid": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "stress_covid_B15000_final_v2.csv",
    "stress_B_2024": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "stress_2024_B15000_final_v2.csv",
    "evt_B": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "cedi_evt_threshold_sensitivity_publication_v2.csv",
    "cedi_AG": ROOT / "outputs" / "tables" / "cedi_robustness_AG_full.csv",
    "pit_tie_detail": ROOT / "outputs" / "pit_ties_audit" / "panelA_cedi_tie_detail.csv",
    "pit_tie_gof": ROOT / "outputs" / "pit_ties_audit" / "panelA_cedi_declipped_observed_gof_sensitivity_v2.csv",
    "marginal_A": ROOT / "outputs" / "original_panel_gate_audit" / "original_panel_marginal_selected_hard_gate.csv",
    "marginal_B": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "marginal_selected.csv",
}


def require_files() -> None:
    missing = [str(p.relative_to(ROOT)) for p in PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen figure inputs:\n  - " + "\n  - ".join(missing)
        )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_pair(x: str) -> str:
    s = (
        str(x)
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace(" ", "")
    )
    labels = {
        "cocoa": "Cocoa",
        "gold": "Gold",
        "brent": "Brent",
        "wti": "WTI",
        "cedi": "Cedi",
    }
    return "-".join(labels.get(part.lower(), part) for part in s.split("-"))


def asset_label(x: str) -> str:
    return {
        "cocoa": "Cocoa",
        "gold": "Gold",
        "brent": "Brent",
        "wti": "WTI",
        "cedi": "Cedi",
    }.get(str(x).lower(), str(x))


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def pseudo_obs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = len(df)
    for c in df.columns:
        out[c] = df[c].rank(method="average") / (n + 1.0)
    return out


def empirical_tail_table(pit: pd.DataFrame) -> pd.DataFrame:
    u = pseudo_obs(pit)
    cols = ["cocoa", "gold", "brent", "wti", "cedi"]
    if not all(c in u.columns for c in cols):
        raise ValueError(f"PIT columns are {list(u.columns)}; expected {cols}")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = f"{asset_label(a)}-{asset_label(b)}"
            for q in Q_LEVELS:
                lower = ((u[a] <= q) & (u[b] <= q)).sum() / (len(u) * q)
                upper = ((u[a] >= 1 - q) & (u[b] >= 1 - q)).sum() / (len(u) * q)
                rows.extend([
                    {"pair": pair, "a": a, "b": b, "q": q, "tail": "lower", "lambda_empirical": lower},
                    {"pair": pair, "a": a, "b": b, "q": q, "tail": "upper", "lambda_empirical": upper},
                ])
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save archival outputs and refresh the PNG used by the LaTeX rebuild."""
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(MANUSCRIPT_FIGURES / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))


def box(ax, xy, wh, title, body="", fontsize=9):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        fill=False,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=fontsize + 1, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.38, body, ha="center", va="center",
                fontsize=fontsize, linespacing=1.25)
    return p


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.0))


def build_stress_summary():
    # Panel A
    a_comb = pd.read_csv(PATHS["stress_A_combined"])
    a_dis = pd.read_csv(PATHS["stress_A_disagg"])
    a_comb["stress"] = "Combined"
    a_dis["stress"] = a_dis["stress_def"].astype(str).str.lower().map(
        {"covid_only": "COVID-19", "2024_only": "2024"}
    )

    # Panel B
    b_files = {
        "Combined": PATHS["stress_B_combined"],
        "COVID-19": PATHS["stress_B_covid"],
        "2024": PATHS["stress_B_2024"],
    }

    rows = []
    for stress, df in [
        ("Combined", a_comb),
        ("COVID-19", a_dis[a_dis["stress"] == "COVID-19"]),
        ("2024", a_dis[a_dis["stress"] == "2024"]),
    ]:
        rows.append({
            "panel": "Panel A",
            "stress": stress,
            "n_tests": len(df),
            "n_stress": np.nan,
            "min_p": df["p_value"].min(),
            "nominal": int(as_bool(df["sig_uncorrected_5pct"]).sum()),
            "bonf": int(as_bool(df["sig_bonferroni_m60"]).sum()),
            "bh": int(as_bool(df["sig_FDR_BH_5pct"]).sum()),
        })

    for stress, path in b_files.items():
        df = pd.read_csv(path)
        rows.append({
            "panel": "Panel B",
            "stress": stress,
            "n_tests": len(df),
            "n_stress": int(df["n_stress"].iloc[0]),
            "min_p": df["p_value"].min(),
            "nominal": int(as_bool(df["uncorrected_reject_5pct"]).sum()),
            "bonf": int(as_bool(df["bonf_reject_5pct"]).sum()),
            "bh": int(as_bool(df["bh_reject_5pct"]).sum()),
        })
    out = pd.DataFrame(rows)
    return out


def strict_freeze_checks(gof_summary: pd.DataFrame, stress_summary: pd.DataFrame) -> None:
    # GOF expected non-rejected counts
    exp_A = {
        "Cocoa-Gold": 0, "Cocoa-Brent": 0, "Cocoa-WTI": 0, "Cocoa-Cedi": 8,
        "Gold-Brent": 0, "Gold-WTI": 0, "Gold-Cedi": 8,
        "Brent-WTI": 0, "Brent-Cedi": 8, "WTI-Cedi": 8,
    }
    exp_B = {
        "Cocoa-Gold": 6, "Cocoa-Brent": 5, "Cocoa-WTI": 5, "Cocoa-Cedi": 8,
        "Gold-Brent": 0, "Gold-WTI": 0, "Gold-Cedi": 8,
        "Brent-WTI": 0, "Brent-Cedi": 8, "WTI-Cedi": 8,
    }
    for pair in PAIRS_ORDER:
        gotA = int(gof_summary.loc[(gof_summary.panel == "Panel A") &
                                   (gof_summary.pair == pair), "nonrejected"].iloc[0])
        gotB = int(gof_summary.loc[(gof_summary.panel == "Panel B") &
                                   (gof_summary.pair == pair), "nonrejected"].iloc[0])
        assert gotA == exp_A[pair], (pair, "Panel A", gotA, exp_A[pair])
        assert gotB == exp_B[pair], (pair, "Panel B", gotB, exp_B[pair])

    # Stress counts are read from the synchronized frozen output tables.
    # Nominal 5% counts are descriptive. The publication-level invariants
    # enforced here are 60 cells per stress family and zero Bonferroni/BH
    # discoveries in both panels.
    for panel in ["Panel A", "Panel B"]:
        for stress in ["Combined", "COVID-19", "2024"]:
            key = (panel, stress)
            row = stress_summary[(stress_summary.panel == panel) &
                                 (stress_summary.stress == stress)].iloc[0]
            assert int(row["n_tests"]) == 60, (key, "n_tests", row["n_tests"])
            assert int(row["bonf"]) == 0, (key, "bonf", row["bonf"])
            assert int(row["bh"]) == 0, (key, "bh", row["bh"])


def figure_01_calendar_workflow():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.02, 0.25), (0.12, 0.50), "Raw sources",
        "Yahoo Finance\ncommodity prices\n\nBank of Ghana\nUSD/GHS")
    box(ax, (0.18, 0.55), (0.16, 0.25), "Panel A",
        "Business-day alignment\n3,008 returns\n2015-01-05 to 2026-07-15\nffill capped at 3 business days",
        fontsize=8.5)
    box(ax, (0.18, 0.18), (0.16, 0.25), "Panel B",
        "Common-observation\nalignment\n2,774 returns\n2015-01-05 to 2026-07-10\nno forward-filled returns",
        fontsize=8.5)
    box(ax, (0.39, 0.25), (0.13, 0.50), "Marginal stage",
        "Fixed 4-model ladder\nhard convergence gate\nadequacy diagnostics\nPIT construction")
    box(ax, (0.56, 0.25), (0.12, 0.50), "Dependence",
        "8 static copula families\nabsolute GOF\nfinite-threshold\nconcentration")
    box(ax, (0.72, 0.25), (0.14, 0.50), "Stress inference",
        "Combined\nCOVID-19\n2024\n\nMBB B=15,000\n60 tests per family")
    box(ax, (0.89, 0.25), (0.095, 0.50), "Interpretation",
        "calendar-robust\ncalendar-sensitive\npanel-specific", fontsize=8)

    # Clean fork from the raw sources into the two co-equal calendar panels.
    ax.plot([0.14, 0.158], [0.50, 0.50], linewidth=1.0, color="black")
    ax.plot([0.158, 0.158], [0.50, 0.675], linewidth=1.0, color="black")
    ax.plot([0.158, 0.158], [0.305, 0.50], linewidth=1.0, color="black")
    arrow(ax, 0.158, 0.675, 0.18, 0.675)
    arrow(ax, 0.158, 0.305, 0.18, 0.305)

    arrow(ax, 0.34, 0.675, 0.39, 0.57)
    arrow(ax, 0.34, 0.305, 0.39, 0.43)
    arrow(ax, 0.52, 0.50, 0.56, 0.50)
    arrow(ax, 0.68, 0.50, 0.72, 0.50)
    arrow(ax, 0.86, 0.50, 0.89, 0.50)

    ax.text(
        0.5, 0.06,
        "Panel A and Panel B are co-equal calendar constructions; neither is treated as the primary or corrected panel.",
        ha="center", va="center", fontsize=9.5, fontweight="bold"
    )
    save_figure(fig, "figure_01_calendar_workflow")


def figure_02_tail_concentration_profiles(tail_A, tail_B):
    chosen = ["Cocoa-Gold", "Cocoa-Brent", "Gold-WTI", "Gold-Cedi"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for k, (ax, pair) in enumerate(zip(axes.flat, chosen)):
        for panel, tab, marker in [("Panel A", tail_A, "o"), ("Panel B", tail_B, "s")]:
            ss = tab[tab["pair"].map(canonical_pair) == canonical_pair(pair)]
            if ss.empty:
                raise ValueError(f"No tail-concentration rows found for {pair} in {panel}")
            for tail, ls in [("lower", "-"), ("upper", "--")]:
                mask = ss["tail"] == tail
                x = ss.loc[mask, "q"].astype(float).to_numpy()
                y = ss.loc[mask, "lambda_empirical"].astype(float).to_numpy()
                order = np.argsort(x)
                ax.plot(x[order], y[order], marker=marker, linestyle=ls,
                        label=f"{panel} {tail}")
        ax.set_title(f"({chr(97 + k)}) {pair}")
        ax.set_xticks(Q_LEVELS, ["0.025", "0.05", "0.10"])
        ax.set_ylim(0, 0.21)
        ax.grid(alpha=0.2)

    axes[1, 0].set_xlabel("Threshold q")
    axes[1, 1].set_xlabel("Threshold q")
    axes[0, 0].set_ylabel("Finite-threshold concentration")
    axes[1, 0].set_ylabel("Finite-threshold concentration")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=4, fontsize=8, frameon=False)

    fig.text(
        0.5, 0.012,
        "Values are empirical finite-threshold concentration measures, not asymptotic tail-dependence coefficients.",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    save_figure(fig, "figure_02_tail_concentration_profiles")


def build_gof_summary():
    a = pd.read_csv(PATHS["gof_A"])
    b = pd.read_csv(PATHS["gof_B"])
    a["pair"] = a["pair"].map(canonical_pair)
    b["pair"] = b["pair"].map(canonical_pair)

    a["nonreject"] = as_bool(a["not_rejected_5pct"])
    b["nonreject"] = as_bool(b["nonreject_5pct"])

    rows = []
    for pair in PAIRS_ORDER:
        rows.append({
            "panel": "Panel A",
            "pair": pair,
            "nonrejected": int(a.loc[a.pair == pair, "nonreject"].sum()),
        })
        rows.append({
            "panel": "Panel B",
            "pair": pair,
            "nonrejected": int(b.loc[b.pair == pair, "nonreject"].sum()),
        })
    return pd.DataFrame(rows)


def figure_03_gof_heatmap(gof_summary):
    mat = np.zeros((len(PAIRS_ORDER), 2), dtype=int)
    for i, pair in enumerate(PAIRS_ORDER):
        for j, panel in enumerate(["Panel A", "Panel B"]):
            mat[i, j] = int(gof_summary.loc[
                (gof_summary.panel == panel) & (gof_summary.pair == pair),
                "nonrejected"
            ].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=8)
    ax.set_xticks([0, 1], ["Panel A", "Panel B"])
    ax.set_yticks(range(len(PAIRS_ORDER)), PAIRS_ORDER)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            ax.text(j, i, str(value), ha="center", va="center",
                    fontweight="bold", color=("white" if value <= 3 else "black"))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Families not rejected (of 8)")
    ax.set_xlabel("Co-equal calendar construction")
    fig.text(
        0.5, 0.01,
        "0 = all eight tested static families rejected; 8 = none rejected at 5%. Non-rejection is not proof of correctness.",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "figure_03_copula_gof_heatmap")


def figure_04_stress_summary(stress):
    order = ["Combined", "COVID-19", "2024"]
    x = np.arange(len(order))
    width = 0.34

    a = stress[stress.panel == "Panel A"].set_index("stress").loc[order]
    b = stress[stress.panel == "Panel B"].set_index("stress").loc[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - width / 2, a["nominal"], width, label="Panel A: nominal p<0.05")
    bars_b = ax.bar(x + width / 2, b["nominal"], width, label="Panel B: nominal p<0.05")

    ax.set_xticks(x, order)
    ax.set_ylabel("Nominal p<0.05 findings out of 60 tests")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")

    for bars in [bars_a, bars_b]:
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.2,
                    f"{int(r.get_height())}/60", ha="center", va="bottom", fontsize=9)

    y_max = max(stress["nominal"]) + 4.0
    ax.set_ylim(0, y_max)
    ax.text(
        0.02, 0.96,
        "Multiplicity-controlled discoveries: 0/60 in every panel × stress family\n"
        "Bonferroni = 0 throughout; BH-FDR = 0 throughout",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fill=False)
    )

    fig.text(
        0.5, 0.01,
        "Nominal counts are descriptive; no stress-calm test survives multiplicity correction in either calendar construction.",
        ha="center", fontsize=9, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "figure_04_stress_summary")


def figure_05_evt():
    d = pd.read_csv(PATHS["evt_B"]).sort_values("quantile")
    metrics = [
        ("xi", "GPD shape ξ"),
        ("VaR99_resid", "VaR99 (standardized-residual units)"),
        ("ES99_resid", "ES99 (standardized-residual units)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for k, (ax, (col, label)) in enumerate(zip(axes, metrics)):
        ax.plot(d["quantile"], d[col], marker="o")
        ax.axvline(0.90, linestyle="--", linewidth=1)
        row90 = d.loc[np.isclose(d["quantile"], 0.90)].iloc[0]
        ax.scatter([0.90], [row90[col]], zorder=3)
        ax.annotate(f"q = 0.90\n{row90[col]:.4f}",
                    (0.90, row90[col]), xytext=(8, 10),
                    textcoords="offset points", fontsize=8)
        ax.set_title(f"({chr(97 + k)})")
        ax.set_xlabel("POT threshold quantile")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)

    fig.text(
        0.5, 0.01,
        "Conditional EVT diagnostics: the selected Cedi marginal reference model does not pass the PIT adequacy gate.",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    save_figure(fig, "figure_S1_cedi_evt_threshold_sensitivity")


def write_cross_panel_table():
    """Write a manuscript-ready cross-panel evidence table as CSV and Markdown."""
    rows = [
        ["Return observations", "3,008", "2,774"],
        ["Return period", "2015-01-05 to 2026-07-15", "2015-01-05 to 2026-07-10"],
        ["Alignment rule", "Business-day alignment; forward fill capped at 3 business days", "Consecutive common actual-source dates; no forward-filled returns"],
        ["Commodity–Cedi static copula GOF", "All four pairs: 8/8 families not rejected", "All four pairs: 8/8 families not rejected"],
        ["Complete eight-family rejection in both panels", "Gold–Brent; Gold–WTI; Brent–WTI", "Gold–Brent; Gold–WTI; Brent–WTI"],
        ["Cocoa–commodity pairs", "Cocoa–Gold, Cocoa–Brent, Cocoa–WTI: 8/8 families rejected", "Cocoa–Gold: 2/8 rejected; Cocoa–Brent: 3/8; Cocoa–WTI: 3/8"],
        ["Multiplicity-corrected stress discoveries", "0 under Combined, COVID-19, and 2024 definitions", "0 under Combined, COVID-19, and 2024 definitions"],
        ["Analytical role", "Co-equal calendar construction", "Co-equal calendar construction"],
    ]
    table = pd.DataFrame(rows, columns=[
        "Feature",
        "Panel A: business-day alignment",
        "Panel B: common-observation alignment",
    ])
    table.to_csv(OUT / "table_01_cross_panel_evidence.csv", index=False)

    md = [
        "# Table 1. Cross-panel evidence summary",
        "",
        "| Feature | Panel A: business-day alignment | Panel B: common-observation alignment |",
        "|---|---|---|",
    ]
    for row in rows:
        md.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    md.extend([
        "",
        "*Note.* The panels are treated as co-equal defensible representations of the observation process. "
        "GOF counts refer to the eight tested static copula families. Non-rejection does not establish model correctness.",
    ])
    (OUT / "table_01_cross_panel_evidence.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return table


def figure_06_cross_panel(gof_summary, stress):
    data = [
        ["Return observations", "3,008", "2,774"],
        ["Return period", "2015-01-05 to 2026-07-15", "2015-01-05 to 2026-07-10"],
        ["Alignment rule", "Business-day; ffill <= 3 business days", "Consecutive common actual-source dates"],
        ["Commodity-Cedi GOF", "All 4 pairs: 8/8 not rejected", "All 4 pairs: 8/8 not rejected"],
        ["Robust 0/8 not rejected", "Gold-Brent; Gold-WTI; Brent-WTI", "Gold-Brent; Gold-WTI; Brent-WTI"],
        ["Cocoa commodity pairs", "0/8 not rejected for all 3 pairs", "6/8, 5/8, 5/8 not rejected"],
        ["Corrected stress discoveries", "0 under all 3 families", "0 under all 3 families"],
        ["Role", "Co-equal panel", "Co-equal panel"],
    ]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    table = ax.table(
        cellText=data,
        colLabels=["Feature", "Panel A: business-day alignment", "Panel B: common-observation alignment"],
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.22, 0.39, 0.39],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
    fig.text(
        0.5, 0.03,
        "The panels are alternative defensible representations of the observation process; neither is designated primary.",
        ha="center", fontsize=10, fontweight="bold"
    )
    save_figure(fig, "cross_panel_evidence_summary")


def figure_07_cedi_AG():
    d = pd.read_csv(PATHS["cedi_AG"])
    order = sorted(d["treatment"].dropna().unique(), key=lambda s: str(s)[0])
    counts = d.groupby("treatment")["sig_nominal_5pct"].apply(lambda x: as_bool(x).sum()).reindex(order)

    # Brent-Cedi q=.05 sensitivity
    d2 = d.copy()
    d2["pair_canon"] = d2["pair"].map(canonical_pair)
    br = d2[(d2.pair_canon == "Brent-Cedi") & np.isclose(d2["q"], 0.05)]
    br = br.set_index("treatment").reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(np.arange(len(order)), counts.to_numpy())
    axes[0].set_xticks(np.arange(len(order)), [x.split("_")[0] for x in order])
    axes[0].set_ylabel("Nominal p<0.05 findings out of 12")
    axes[0].set_xlabel("Treatment")
    axes[0].set_title("Nominal diagnostic counts")
    axes[0].grid(axis="y", alpha=0.2)
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.08, str(int(v)), ha="center", fontsize=9)

    axes[1].plot(np.arange(len(order)), br["difference"].to_numpy(), marker="o")
    axes[1].axhline(0, linewidth=1)
    axes[1].set_xticks(np.arange(len(order)), [x.split("_")[0] for x in order])
    axes[1].set_ylabel("Stress − calm lower-tail difference")
    axes[1].set_xlabel("Treatment")
    axes[1].set_title("Brent-Cedi, q = 0.05")
    axes[1].grid(alpha=0.2)
    for i, (_, row) in enumerate(br.iterrows()):
        p = row["p_value"]
        if pd.notna(p):
            axes[1].annotate(f"p={p:.3f}", (i, row["difference"]),
                             xytext=(0, 8), textcoords="offset points",
                             ha="center", fontsize=7)

    fig.text(
        0.5, 0.01,
        "A-G are uncorrected Panel A diagnostics. Weekly treatment D produces zero nominal findings; these are not headline discoveries.",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    save_figure(fig, "figure_S2_cedi_AG_sensitivity")


def figure_08_pit_clipping():
    tie = pd.read_csv(PATHS["pit_tie_detail"])
    gof = pd.read_csv(PATHS["pit_tie_gof"])
    family_labels = {
        "gaussian": "Gaussian",
        "clayton": "Clayton",
        "frank": "Frank",
        "gumbel": "Gumbel",
        "survival_gumbel": "Survival Gumbel",
        "t": "Student-t",
        "student_t": "Student-t",
    }
    commodity_labels = gof["commodity"].astype(str).map(asset_label)
    gof["label"] = commodity_labels + "–" + gof["family"].astype(str).map(
        lambda x: family_labels.get(x, x.replace("_", " ").title())
    )

    top = gof.sort_values("abs_Sn_change", ascending=False).head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].axis("off")
    dates = "\n".join(
        f"{r.date}: PIT = {r.pit_value:.0e}, average-rank u = {r.average_rank_percentile:.9f}"
        for r in tie.itertuples()
    )
    axes[0].text(
        0.02, 0.92,
        "Observed clipping tie\n\n"
        f"Three observations at PIT = {tie['pit_value'].iloc[0]:.0e}\n\n"
        f"{dates}\n\n"
        "Recovered residual order gives ranks 3, 2, 1.\n"
        "Only two pseudo-observations change.\n"
        "Tail membership changes at q = 0.025, 0.05, 0.10: 0.",
        va="top", fontsize=10,
        bbox=dict(boxstyle="round", fill=False)
    )

    axes[1].barh(top["label"], top["abs_Sn_change"])
    axes[1].set_xlabel("Absolute change in observed GOF statistic")
    axes[1].set_title("Largest observed GOF changes after de-clipping")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    max_change = gof["abs_Sn_change"].max()
    axes[1].text(
        0.98, 0.03,
        f"Maximum |ΔSn| = {max_change:.6f}\nCocoa–Cedi Frank |ΔSn| = 0",
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=9
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save_figure(fig, "figure_S3_pit_clipping_diagnostic")


def figure_09_study_design():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    titles = [
        ("Research questions", "RQ1 stress-tail differences\nRQ2 static copula adequacy\nRQ3 Cedi sensitivity"),
        ("Two co-equal panels", "Panel A business-day\nPanel B common-observation"),
        ("Marginal stage", "4-model ladder\nconvergence + adequacy gate\nstandardized residuals + PITs"),
        ("Dependence stage", "8 static copulas\nabsolute GOF\nfinite-threshold concentration"),
        ("Inference", "3 stress families\nMBB B=15,000\nBonferroni + BH"),
        ("Cross-panel synthesis", "calendar-robust\ncalendar-sensitive\npanel-specific"),
    ]
    x0s = np.linspace(0.02, 0.83, len(titles))
    w = 0.14
    for i, (title, body) in enumerate(titles):
        box(ax, (x0s[i], 0.27), (w, 0.46), title, body, fontsize=8)
        if i < len(titles) - 1:
            arrow(ax, x0s[i] + w, 0.50, x0s[i+1], 0.50)

    ax.text(
        0.5, 0.10,
        "Design principle: conclusions are promoted only when their evidential status is explicit under both defensible calendar constructions.",
        ha="center", fontsize=10, fontweight="bold"
    )
    save_figure(fig, "repository_study_design_workflow")


def figure_10_findings_summary(gof_summary, stress):
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cards = [
        (0.03, 0.30, 0.20, 0.52,
         "1  Data architecture",
         "Panel A\n3,008 returns\nbusiness-day alignment\n\n"
         "Panel B\n2,774 returns\ncommon-observation alignment\n\n"
         "Co-equal constructions"),
        (0.28, 0.30, 0.20, 0.52,
         "2  Static copula GOF",
         "Robust 8/8 rejection\nGold–Brent\nGold–WTI\nBrent–WTI\n\n"
         "Robust 0/8 rejection\nall four commodity–Cedi pairs\n\n"
         "Calendar-sensitive\nthree Cocoa–commodity pairs"),
        (0.53, 0.30, 0.20, 0.52,
         "3  Stress inference",
         f"Nominal p < 0.05\nCombined: "
         f"{int(stress[(stress.panel == 'Panel A') & (stress.stress == 'Combined')]['nominal'].iloc[0])}/60 vs "
         f"{int(stress[(stress.panel == 'Panel B') & (stress.stress == 'Combined')]['nominal'].iloc[0])}/60\n"
         f"COVID-19: {int(stress[(stress.panel == 'Panel A') & (stress.stress == 'COVID-19')]['nominal'].iloc[0])}/60 vs "
         f"{int(stress[(stress.panel == 'Panel B') & (stress.stress == 'COVID-19')]['nominal'].iloc[0])}/60\n"
         f"2024: {int(stress[(stress.panel == 'Panel A') & (stress.stress == '2024')]['nominal'].iloc[0])}/60 vs "
         f"{int(stress[(stress.panel == 'Panel B') & (stress.stress == '2024')]['nominal'].iloc[0])}/60\n\n"
         "Bonferroni: 0/60 throughout\nBH-FDR: 0/60 throughout"),
        (0.78, 0.30, 0.19, 0.52,
         "4  Interpretation",
         "Some static GOF results are\ncalendar-sensitive.\n\n"
         "No stress–calm test survives\nmultiplicity correction.\n\n"
         "Cedi EVT remains conditional\non marginal inadequacy.\n\n"
         "No monthly transmission\nclaim is retained."),
    ]

    for x, y, w, h, title, body in cards:
        box(ax, (x, y), (w, h), title, body, fontsize=8.5)

    arrow(ax, 0.23, 0.56, 0.28, 0.56)
    arrow(ax, 0.48, 0.56, 0.53, 0.56)
    arrow(ax, 0.73, 0.56, 0.78, 0.56)

    ax.text(
        0.5, 0.13,
        "Inference principle: cross-panel robustness + absolute model adequacy + multiplicity-controlled testing.",
        ha="center", va="center", fontsize=10.5, fontweight="bold"
    )
    ax.text(
        0.5, 0.065,
        "Calendar construction changes some dependence-model adequacy classifications, but not the multiplicity-controlled stress-inference conclusion.",
        ha="center", va="center", fontsize=9.2
    )

    save_figure(fig, "graphical_abstract_verified_findings")


def write_manifest_and_captions(stress_summary, gof_summary):
    manifest = pd.DataFrame([
        ["Figure 1", "figure_01_calendar_workflow", "main candidate", "conceptual + frozen panel geometry"],
        ["Figure 2", "figure_02_tail_concentration_profiles", "main candidate", "Panel A/B PIT files"],
        ["Figure 3", "figure_03_copula_gof_heatmap", "main candidate", "Panel A/B GOF outputs"],
        ["Figure 4", "figure_04_stress_summary", "main candidate", "Panel A/B B=15,000 stress outputs"],
        ["Figure S1", "figure_S1_cedi_evt_threshold_sensitivity", "supplement", "Panel B EVT threshold output"],
        ["Table 1 preview", "cross_panel_evidence_summary", "internal preview; use table_01_cross_panel_evidence.md in manuscript", "frozen cross-panel evidence"],
        ["Figure S2", "figure_S2_cedi_AG_sensitivity", "supplement", "Panel A A-G diagnostic output"],
        ["Figure S3", "figure_S3_pit_clipping_diagnostic", "supplement", "Panel A clipping/tie audit"],
        ["Repository", "repository_study_design_workflow", "repository only; overlaps Figure 1", "frozen study design"],
        ["Graphical abstract", "graphical_abstract_verified_findings", "graphical abstract", "frozen headline findings"],
    ], columns=["figure", "stem", "suggested_role", "source"])
    manifest.to_csv(OUT / "figure_manifest.csv", index=False)

    hashes = []
    for name, path in PATHS.items():
        hashes.append(f"{name}\t{path.relative_to(ROOT)}\t{sha256(path)}")
    (OUT / "figure_input_hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    captions = """# Candidate figure captions

**Figure 1. Data and calendar-construction workflow.** The two co-equal panels represent alternative defensible treatments of nonsynchronous source calendars. Panel A preserves the documented business-day alignment with limited forward filling; Panel B uses consecutive common actual-source observation dates.

**Figure 2. Empirical finite-threshold tail-concentration profiles.** Lower- and upper-tail concentration at q = 0.025, 0.05, and 0.10 for selected pairs under both calendar constructions. The measures are finite-threshold empirical concentrations and are not interpreted as asymptotic tail-dependence coefficients.

**Figure 3. Static copula goodness-of-fit comparison.** Number of the eight tested static copula families not rejected at the 5% level for each pair and panel. Zero indicates complete eight-family rejection; eight indicates no family is rejected. Non-rejection does not establish model correctness.

**Figure 4. Stress-calm tail-concentration tests: nominal versus adjusted evidence.** Nominal 5% stress-calm findings are shown for the combined, COVID-19, and 2024 stress definitions. Each panel-by-stress family contains 60 tests. Nominal counts are descriptive; no test survives Bonferroni or Benjamini-Hochberg correction in either panel.

**Figure S1. Panel B Cedi EVT threshold sensitivity.** Generalized Pareto shape, 99% VaR, and 99% expected shortfall across POT thresholds. Values are conditional diagnostics because the selected Cedi marginal reference model fails the PIT adequacy gate.

**Table 1. Cross-panel evidence summary.** Manuscript-ready table exported separately as `table_01_cross_panel_evidence.md` and `.csv`.

**Figure S2. Panel A Cedi A-G sensitivity analysis.** Nominal diagnostic counts across seven alternative Cedi treatments and the Brent-Cedi q = 0.05 lower-tail stress-calm difference. These are uncorrected sensitivity diagnostics rather than headline discoveries.

**Figure S3. Panel A Cedi PIT-clipping diagnostic.** The three-way clipping tie at 1e-6, recovered residual rank order, and observed GOF sensitivity. No prespecified tail-set membership changes, and observed GOF changes are negligible.

**Repository figure. Study design workflow overview.** Relationship between the research questions, co-equal calendar panels, marginal modeling, copula adequacy, stress inference, and cross-panel classification.

**Graphical abstract. Cross-panel evidence from two co-equal calendar constructions.** Compact synthesis of the data architecture, robust and calendar-sensitive static-copula GOF findings, multiplicity-controlled stress inference, and diagnostic limitations.
"""
    (OUT / "figure_captions.md").write_text(captions, encoding="utf-8")


def main():
    require_files()

    pit_A = pd.read_csv(PATHS["pit_A"], index_col=0, parse_dates=True)
    pit_B = pd.read_csv(PATHS["pit_B"], index_col=0, parse_dates=True)

    assert len(pit_A) == 3007, f"Unexpected Panel A PIT rows: {len(pit_A)}"
    assert len(pit_B) == 2773, f"Unexpected Panel B PIT rows: {len(pit_B)}"

    tail_A = empirical_tail_table(pit_A)
    tail_B = empirical_tail_table(pit_B)

    gof_summary = build_gof_summary()
    stress_summary = build_stress_summary()
    strict_freeze_checks(gof_summary, stress_summary)

    print("Observed nominal stress counts from synchronized source tables:")
    for _, r in stress_summary.sort_values(["panel", "stress"]).iterrows():
        print(f"  {r['panel']} | {r['stress']}: nominal={int(r['nominal'])}/60, "
              f"Bonferroni={int(r['bonf'])}/60, BH={int(r['bh'])}/60")

    tail_A.assign(panel="Panel A").to_csv(OUT / "derived_tail_concentration_panelA.csv", index=False)
    tail_B.assign(panel="Panel B").to_csv(OUT / "derived_tail_concentration_panelB.csv", index=False)
    gof_summary.to_csv(OUT / "derived_gof_nonrejection_counts.csv", index=False)
    stress_summary.to_csv(OUT / "derived_stress_summary_counts.csv", index=False)

    figure_01_calendar_workflow()
    figure_02_tail_concentration_profiles(tail_A, tail_B)
    figure_03_gof_heatmap(gof_summary)
    figure_04_stress_summary(stress_summary)
    figure_05_evt()
    write_cross_panel_table()
    figure_06_cross_panel(gof_summary, stress_summary)
    figure_07_cedi_AG()
    figure_08_pit_clipping()
    figure_09_study_design()
    figure_10_findings_summary(gof_summary, stress_summary)

    write_manifest_and_captions(stress_summary, gof_summary)

    print(f"Created publication figure candidate suite in: {OUT}")
    print("Freeze checks passed.")
    print("Generated main figures, supplementary figures, graphical abstract, repository workflow, and cross-panel manuscript table.")
    print(f"Refreshed LaTeX PNG assets in: {MANUSCRIPT_FIGURES}")
    print("No model fitting or bootstrap computation was performed.")


if __name__ == "__main__":
    main()
