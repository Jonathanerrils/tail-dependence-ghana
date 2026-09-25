"""Cheap GOF implementation-equivalence audit.

Purpose
-------
Before any expensive GOF rerun, verify whether the historical Panel A
publication GOF and the Panel B publication GOF used the same estimator,
statistic, p-value convention, and input state.

This script performs ZERO bootstrap replications.

Run from repository root:
    python scripts/32_gof_implementation_equivalence_audit.py

It writes only to:
    outputs/gof_equivalence_audit/
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk import copulas, gof  # noqa: E402

PANEL_A_PIT = ROOT / "outputs" / "tables" / "_pit_real.csv"
PANEL_B_PIT = (
    ROOT / "outputs" / "publication_rebuild_v2" / "tables"
    / "_pit_publication_v2.csv"
)
OUT = ROOT / "outputs" / "gof_equivalence_audit"
OUT.mkdir(parents=True, exist_ok=True)

DETAIL = OUT / "observed_estimator_comparison_panelA.csv"
SUMMARY = OUT / "gof_implementation_equivalence_summary.json"
HASHES = OUT / "gof_implementation_equivalence_hashes.txt"

FAMILIES = [
    "gaussian", "t", "clayton", "gumbel",
    "frank", "joe", "survival_clayton", "survival_gumbel",
]

EXPECTED_PANEL_A_SHAPE = (3007, 5)
EXPECTED_PANEL_A_START = "2015-01-06"
EXPECTED_PANEL_A_END = "2026-07-15"

EXPECTED_PANEL_B_SHA = (
    "9b322bda7c4e63534da14d1a7e360e758762e98cb39101cf329cdceb5afc216f"
)
EXPECTED_PANEL_B_SHAPE = (2773, 5)
EXPECTED_PANEL_B_START = "2015-01-06"
EXPECTED_PANEL_B_END = "2026-07-10"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pit(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def pair_pseudo(df: pd.DataFrame, a: str, b: str):
    po = copulas.pseudo_obs(df[[a, b]])
    return po[:, 0], po[:, 1]


def param_distance(old: dict, new: dict):
    keys = sorted(set(old) | set(new))
    vals = {}
    max_abs = 0.0
    for k in keys:
        ov = float(old[k]) if k in old else np.nan
        nv = float(new[k]) if k in new else np.nan
        d = abs(ov - nv) if np.isfinite(ov) and np.isfinite(nv) else np.nan
        vals[f"old_{k}"] = ov
        vals[f"new_{k}"] = nv
        vals[f"absdiff_{k}"] = d
        if np.isfinite(d):
            max_abs = max(max_abs, d)
    return vals, max_abs


if not PANEL_A_PIT.exists():
    raise FileNotFoundError(PANEL_A_PIT)

panel_a = read_pit(PANEL_A_PIT)
panel_a_sha = sha256(PANEL_A_PIT)

print("=== GOF IMPLEMENTATION-EQUIVALENCE AUDIT ===")
print("No bootstrap computation will be run.")
print()
print("--- Panel A PIT state ---")
print(f"path:   {PANEL_A_PIT.relative_to(ROOT)}")
print(f"SHA256: {panel_a_sha}")
print(f"shape:  {panel_a.shape}")
print(
    f"period: {panel_a.index.min().date()} "
    f"to {panel_a.index.max().date()}"
)

panel_a_current = bool(
    tuple(panel_a.shape) == EXPECTED_PANEL_A_SHAPE
    and str(panel_a.index.min().date()) == EXPECTED_PANEL_A_START
    and str(panel_a.index.max().date()) == EXPECTED_PANEL_A_END
)
print(f"matches current audited Panel A PIT geometry: {panel_a_current}")
print()

panel_b_info = None
if PANEL_B_PIT.exists():
    panel_b = read_pit(PANEL_B_PIT)
    panel_b_sha = sha256(PANEL_B_PIT)
    panel_b_current = bool(
        panel_b_sha == EXPECTED_PANEL_B_SHA
        and tuple(panel_b.shape) == EXPECTED_PANEL_B_SHAPE
        and str(panel_b.index.min().date()) == EXPECTED_PANEL_B_START
        and str(panel_b.index.max().date()) == EXPECTED_PANEL_B_END
    )
    panel_b_info = {
        "sha256": panel_b_sha,
        "shape": list(panel_b.shape),
        "period": [
            str(panel_b.index.min().date()),
            str(panel_b.index.max().date()),
        ],
        "freeze_check_passed": panel_b_current,
    }
    print("--- Panel B PIT state ---")
    print(f"SHA256: {panel_b_sha}")
    print(f"shape:  {panel_b.shape}")
    print(
        f"period: {panel_b.index.min().date()} "
        f"to {panel_b.index.max().date()}"
    )
    print(f"freeze check: {panel_b_current}")
    print()

# Static code inspection.
estimate_src = inspect.getsource(gof.estimate)
gof_test_src = inspect.getsource(gof.gof_test)

old_uses_tau = "kendalltau" in estimate_src
old_t_uses_grid = "NU_GRID" in estimate_src
old_half_correction = "exceed + 0.5" in gof_test_src
old_uses_gof_estimate_bootstrap = "th_b = estimate(" in gof_test_src

print("--- Static implementation comparison ---")
print(
    "Historical gof.gof_test estimator: "
    "gof.estimate()"
)
print(
    "Panel B Stage 2 estimator: "
    "copulas.FAMILIES[family](u, v)"
)
print(f"gof.estimate uses Kendall tau: {old_uses_tau}")
print(f"t estimator uses fixed nu grid: {old_t_uses_grid}")
print(
    "gof.gof_test bootstrap refits with gof.estimate: "
    f"{old_uses_gof_estimate_bootstrap}"
)
print(
    "gof.gof_test p-value uses +0.5 correction: "
    f"{old_half_correction}"
)
print(
    "Panel B Stage 2 p-value convention: "
    "(exceed + 1)/(B + 1)"
)
print()

# Quantify observed-data estimator differences on Panel A.
pairs = []
cols = list(panel_a.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((cols[i], cols[j]))

rows = []
for a, b in pairs:
    u, v = pair_pseudo(panel_a, a, b)
    for fam in FAMILIES:
        old_params = gof.estimate(fam, u, v)
        new_fit = copulas.FAMILIES[fam](u, v)
        new_params = new_fit.params

        pvals, max_abs = param_distance(old_params, new_params)

        old_sn = gof.cvm_rosenblatt(
            u, v, fam, old_params, grid_size=50
        )
        new_sn = gof.cvm_rosenblatt(
            u, v, fam, new_params, grid_size=50
        )

        same_params = True
        for k in set(old_params) | set(new_params):
            if k not in old_params or k not in new_params:
                same_params = False
                break
            if not np.isclose(
                float(old_params[k]),
                float(new_params[k]),
                rtol=1e-7,
                atol=1e-9,
            ):
                same_params = False
                break

        rows.append({
            "a": a,
            "b": b,
            "family": fam,
            "same_observed_parameters": same_params,
            "max_abs_parameter_difference": max_abs,
            "Sn_old_gof_estimator": old_sn,
            "Sn_new_copulas_estimator": new_sn,
            "abs_Sn_difference": abs(old_sn - new_sn),
            **pvals,
        })

detail = pd.DataFrame(rows)
detail.to_csv(DETAIL, index=False, float_format="%.12g")

family_summary = []
for fam, g in detail.groupby("family", sort=False):
    family_summary.append({
        "family": fam,
        "cells": int(len(g)),
        "same_parameter_cells": int(g["same_observed_parameters"].sum()),
        "different_parameter_cells": int(
            (~g["same_observed_parameters"]).sum()
        ),
        "max_abs_Sn_difference": float(g["abs_Sn_difference"].max()),
        "mean_abs_Sn_difference": float(g["abs_Sn_difference"].mean()),
    })

fam_df = pd.DataFrame(family_summary)

# The two algorithms are materially non-equivalent if:
# - the fitting estimator differs in any observed cell, or
# - the p-value convention differs.
material_mismatch = bool(
    (~detail["same_observed_parameters"]).any()
    or old_half_correction
)

summary = {
    "panel_A": {
        "sha256": panel_a_sha,
        "shape": list(panel_a.shape),
        "period": [
            str(panel_a.index.min().date()),
            str(panel_a.index.max().date()),
        ],
        "matches_current_audited_panel_A_PIT_geometry": panel_a_current,
    },
    "panel_B": panel_b_info,
    "historical_panel_A_GOF": {
        "estimator": "gof.estimate",
        "uses_kendall_tau_for_multiple_families": old_uses_tau,
        "t_nu_fixed_grid": old_t_uses_grid,
        "bootstrap_refit_uses_same_gof_estimate": old_uses_gof_estimate_bootstrap,
        "p_value_formula": "(exceed + 0.5)/(B + 1)",
    },
    "panel_B_stage2_GOF": {
        "estimator": "copulas.FAMILIES pseudo-MLE / MLE fit functions",
        "bootstrap_refit": "same copulas.FAMILIES estimator",
        "p_value_formula": "(exceed + 1)/(B + 1)",
    },
    "family_observed_parameter_comparison": family_summary,
    "material_implementation_mismatch": material_mismatch,
    "bootstrap_rerun_performed": False,
    "recommended_action": (
        "Rebuild Panel A publication GOF with the exact Panel B Stage 2 "
        "estimation/bootstrap algorithm before making cross-panel GOF claims."
        if material_mismatch
        else
        "No estimator mismatch found; no Panel A GOF rerun required on this basis."
    ),
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

with HASHES.open("w", encoding="utf-8", newline="\n") as f:
    for p in [
        PANEL_A_PIT,
        ROOT / "src" / "tailrisk" / "gof.py",
        ROOT / "src" / "tailrisk" / "copulas.py",
        DETAIL,
        SUMMARY,
    ]:
        if p.exists():
            f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")
    if PANEL_B_PIT.exists():
        f.write(
            f"{sha256(PANEL_B_PIT)}  "
            f"{PANEL_B_PIT.relative_to(ROOT)}\n"
        )

print("=== OBSERVED-ESTIMATOR COMPARISON BY FAMILY ===")
print(fam_df.to_string(index=False))
print()
print("=== AUDIT VERDICT ===")
print(
    "MATERIAL MISMATCH"
    if material_mismatch
    else "NO MATERIAL MISMATCH DETECTED"
)
print(summary["recommended_action"])
print()
print(f"detail:  {DETAIL.relative_to(ROOT)}")
print(f"summary: {SUMMARY.relative_to(ROOT)}")
print(f"hashes:  {HASHES.relative_to(ROOT)}")
print("=== GOF IMPLEMENTATION-EQUIVALENCE AUDIT COMPLETE ===")
