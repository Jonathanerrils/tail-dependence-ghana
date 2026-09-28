"""Audit the canonical monthly transmission regression output.

This script does not rerun or overwrite the regressions. It:
- verifies the expected 9 coefficient tests are present,
- applies Bonferroni and Benjamini-Hochberg correction across those 9 tests,
- writes a clean audit table and compact summary.

Run from repository root:
    python scripts/22_transmission_audit.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"

SOURCE = TAB / "transmission_regressions.csv"
OUT = TAB / "transmission_regressions_audited.csv"
SUMMARY = TAB / "transmission_audit_summary.txt"

ALPHA = 0.05
M = 9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_reject(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float | None]:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    crit = alpha * np.arange(1, m + 1) / m

    passing = np.where(sorted_p <= crit)[0]
    reject = np.zeros(m, dtype=bool)

    if len(passing) == 0:
        return reject, None

    k = passing.max()
    cutoff = float(sorted_p[k])
    reject = pvals <= cutoff
    return reject, cutoff


if not SOURCE.exists():
    raise FileNotFoundError(f"Missing: {SOURCE}")

df = pd.read_csv(SOURCE)

required = {"target", "n", "regressor", "coef", "hac_p", "r2"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if len(df) != M:
    raise ValueError(f"Expected {M} coefficient tests, found {len(df)}")

if df[["target", "regressor"]].duplicated().any():
    dup = df.loc[df[["target", "regressor"]].duplicated(keep=False),
                 ["target", "regressor"]]
    raise ValueError(f"Duplicate target/regressor rows found:\n{dup}")

p = df["hac_p"].astype(float).to_numpy()

bonf_threshold = ALPHA / M
df["sig_nominal_5pct"] = p <= ALPHA
df["sig_bonferroni_5pct"] = p <= bonf_threshold

bh_rej, bh_cutoff = bh_reject(p, ALPHA)
df["sig_bh_fdr_5pct"] = bh_rej

# Bonferroni-adjusted p-values for transparency
df["p_bonf_adjusted"] = np.minimum(p * M, 1.0)

# Standard BH adjusted p-values
order = np.argsort(p)
ranked = p[order]
adj_sorted = ranked * M / np.arange(1, M + 1)
adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
adj_sorted = np.minimum(adj_sorted, 1.0)
bh_adj = np.empty(M)
bh_adj[order] = adj_sorted
df["p_bh_adjusted"] = bh_adj

df.to_csv(
    OUT,
    index=False,
    float_format="%.12f",
    lineterminator="\n",
    encoding="utf-8",
)

print("=== TRANSMISSION AUDIT ===")
print(f"source SHA256: {sha256(SOURCE)}")
print(f"tests: {len(df)}")
print(f"Bonferroni threshold: {bonf_threshold:.12f}")
print(f"BH cutoff: {'none' if bh_cutoff is None else f'{bh_cutoff:.12f}'}")
print()

show = df[
    [
        "target", "n", "regressor", "coef", "hac_p", "r2",
        "sig_nominal_5pct", "sig_bonferroni_5pct", "sig_bh_fdr_5pct",
        "p_bonf_adjusted", "p_bh_adjusted",
    ]
].sort_values("hac_p")

print(show.to_string(index=False))

n_nom = int(df["sig_nominal_5pct"].sum())
n_bonf = int(df["sig_bonferroni_5pct"].sum())
n_bh = int(df["sig_bh_fdr_5pct"].sum())

min_row = df.loc[df["hac_p"].idxmin()]

summary = (
    f"Transmission audit\n"
    f"==================\n"
    f"Source SHA256: {sha256(SOURCE)}\n"
    f"Tests: {M}\n"
    f"Nominal 5% discoveries: {n_nom}/{M}\n"
    f"Bonferroni discoveries: {n_bonf}/{M}\n"
    f"BH-FDR discoveries: {n_bh}/{M}\n"
    f"Minimum p-value: {float(min_row['hac_p']):.12f}\n"
    f"Minimum-p cell: {min_row['target']} ~ {min_row['regressor']}\n"
    f"Bonferroni threshold: {bonf_threshold:.12f}\n"
    f"BH cutoff: {'none' if bh_cutoff is None else f'{bh_cutoff:.12f}'}\n"
)

SUMMARY.write_text(summary, encoding="utf-8")

print()
print(f"Nominal 5% discoveries: {n_nom}/{M}")
print(f"Bonferroni discoveries: {n_bonf}/{M}")
print(f"BH-FDR discoveries:     {n_bh}/{M}")
print()
print(f"Wrote: {OUT.relative_to(ROOT)}")
print(f"Wrote: {SUMMARY.relative_to(ROOT)}")
print("=== TRANSMISSION AUDIT COMPLETE ===")
