"""Audit data provenance, calendar alignment, carry-forward filling, and key edge cases.

This script is non-destructive. It reads the CURRENT local raw/processed data
and writes audit-only tables under outputs/tables/.

Run from repository root:
    python scripts/23_data_provenance_audit.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

YRAW = RAW / "prices_yahoo_raw.csv"
BOG = RAW / "bog" / "bog_historical_interbank_fx_usdghs.csv"
PRICES = PROC / "prices_real.csv"
RETURNS = PROC / "returns_real.csv"
MACRO = PROC / "ghana_macro_real.csv"

OUT_SUM = TAB / "data_provenance_audit_summary.csv"
OUT_FILL = TAB / "data_provenance_fill_audit.csv"
OUT_WTI = TAB / "wti_apr2020_data_audit.csv"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def longest_true_run(s: pd.Series) -> int:
    s = s.fillna(False).astype(bool)
    if not s.any():
        return 0
    groups = (s != s.shift()).cumsum()
    return int(s.groupby(groups).sum().max())

for p in [YRAW, BOG, PRICES, RETURNS]:
    if not p.exists():
        raise FileNotFoundError(p)

TAB.mkdir(parents=True, exist_ok=True)

yraw = pd.read_csv(YRAW, index_col=0, parse_dates=True).sort_index()
prices = pd.read_csv(PRICES, index_col=0, parse_dates=True).sort_index()
rets = pd.read_csv(RETURNS, index_col=0, parse_dates=True).sort_index()
bog = pd.read_csv(BOG, parse_dates=["date"]).sort_values("date")
bog = bog.drop_duplicates("date", keep="first").set_index("date")["mid_rate"].astype(float)

print("=== DATA PROVENANCE AUDIT ===")
for p in [YRAW, BOG, PRICES, RETURNS]:
    print(f"{p.relative_to(ROOT)} SHA256: {sha256(p)}")
print(f"Canonical prices:  {prices.shape}, {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Canonical returns: {rets.shape}, {rets.index.min().date()} to {rets.index.max().date()}")
print()

summary_rows = []

# Canonical return descriptives relevant to data construction.
for c in rets.columns:
    x = rets[c].dropna().astype(float)
    nz = x[x != 0]
    summary_rows.append({
        "series": c,
        "n_returns": len(x),
        "zero_returns": int((x == 0).sum()),
        "zero_return_pct": 100 * float((x == 0).mean()),
        "nonzero_abs_le_0_5pctpt": int((nz.abs() <= 0.5).sum()),
        "nonzero_abs_le_0_5pctpt_pct": 100 * float((nz.abs() <= 0.5).mean()) if len(nz) else np.nan,
        "min_return_pct": float(x.min()),
        "max_return_pct": float(x.max()),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUM, index=False, float_format="%.12f", lineterminator="\n")

print("=== CANONICAL RETURN DESCRIPTIVES ===")
print(summary.to_string(index=False))
print()

# Count dates in the canonical price panel that were not exact source observations
# and therefore entered through short-gap carry-forward/calendar alignment.
fill_rows = []
for c in ["cocoa", "gold", "brent", "wti"]:
    if c not in yraw.columns or c not in prices.columns:
        continue
    src = yraw[c].reindex(prices.index)
    exact_valid = src.notna() & (src > 0)
    carried = prices[c].notna() & ~exact_valid
    fill_rows.append({
        "series": c,
        "canonical_price_days": int(prices[c].notna().sum()),
        "exact_positive_source_days": int(exact_valid.sum()),
        "carried_forward_or_aligned_days": int(carried.sum()),
        "carried_forward_pct": 100 * float(carried.mean()),
        "longest_consecutive_carried_run": longest_true_run(carried),
    })

# BoG exact-day comparison
bog_aligned = bog.reindex(prices.index)
exact_bog = bog_aligned.notna() & (bog_aligned > 0)
carried_bog = prices["cedi"].notna() & ~exact_bog
fill_rows.append({
    "series": "cedi",
    "canonical_price_days": int(prices["cedi"].notna().sum()),
    "exact_positive_source_days": int(exact_bog.sum()),
    "carried_forward_or_aligned_days": int(carried_bog.sum()),
    "carried_forward_pct": 100 * float(carried_bog.mean()),
    "longest_consecutive_carried_run": longest_true_run(carried_bog),
})

fill = pd.DataFrame(fill_rows)
fill.to_csv(OUT_FILL, index=False, float_format="%.12f", lineterminator="\n")
print("=== SOURCE-OBSERVATION VS CARRY-FORWARD COUNTS ===")
print(fill.to_string(index=False))
print()

# Non-positive source observations
print("=== NON-POSITIVE RAW MARKET PRICES ===")
for c in ["cocoa", "gold", "brent", "wti"]:
    if c in yraw:
        bad = yraw[c].dropna()
        bad = bad[bad <= 0]
        if len(bad):
            print(c, {str(k.date()): float(v) for k, v in bad.items()})
        else:
            print(c, "none")
print()

# WTI April 2020 edge case
idx = pd.date_range("2020-04-16", "2020-04-23", freq="D")
wti_audit = pd.DataFrame(index=idx)
wti_audit["raw_wti"] = yraw["wti"].reindex(idx)
wti_audit["processed_wti_price"] = prices["wti"].reindex(idx)
wti_audit["processed_wti_return_pct"] = rets["wti"].reindex(idx)
wti_audit.to_csv(OUT_WTI, index_label="date", float_format="%.12f", lineterminator="\n")
print("=== WTI APRIL 2020 EDGE CASE ===")
print(wti_audit.to_string())
print()

# Reconstruct daily Cedi orientation under the documented common-calendar rule.
b = bog.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq="B")).ffill(limit=3)
bret = 100 * np.log(b).diff()
expected = (-bret).reindex(rets.index)
both = pd.concat([expected.rename("expected"), rets["cedi"].rename("processed")], axis=1).dropna()
diff = (both["expected"] - both["processed"]).abs()
print("=== CEDI ORIENTATION CHECK ===")
print(f"n common: {len(both)}")
print(f"correlation expected vs processed: {both['expected'].corr(both['processed']):.12f}")
print(f"max absolute difference: {diff.max():.12g}")
print()

# End-point freeze check
print("=== SAMPLE ENDPOINT CHECK ===")
print(f"Yahoo raw max date: {yraw.index.max().date()}")
print(f"BoG source max date: {bog.index.max().date()}")
print(f"Canonical processed max date: {rets.index.max().date()}")
if yraw.index.max() > rets.index.max() or bog.index.max() > rets.index.max():
    print("NOTE: at least one source extends beyond the canonical sample endpoint.")
    print("The publication pipeline must enforce the endpoint explicitly rather than rely on source availability.")
else:
    print("No source extends beyond the current canonical endpoint.")
print()

if MACRO.exists():
    macro = pd.read_csv(MACRO, index_col=0, parse_dates=True)
    print("=== MONTHLY MACRO PANEL ===")
    print(f"shape: {macro.shape}; {macro.index.min().date()} to {macro.index.max().date()}")
    for c in macro.columns:
        z = macro[c]
        print(f"{c}: nonmissing={int(z.notna().sum())}, zeros={int((z == 0).sum())}")
    print(f"{MACRO.relative_to(ROOT)} SHA256: {sha256(MACRO)}")
    print()

print("=== AUDIT OUTPUTS ===")
print(OUT_SUM.relative_to(ROOT))
print(OUT_FILL.relative_to(ROOT))
print(OUT_WTI.relative_to(ROOT))
print("=== DATA PROVENANCE AUDIT COMPLETE ===")
