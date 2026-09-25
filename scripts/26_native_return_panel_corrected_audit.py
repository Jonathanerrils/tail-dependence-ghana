"""Corrected audit: compute returns on each series' own observed source dates.

NON-DESTRUCTIVE. This fixes an important detail in script 25:
ordinary source-missing dates are removed BEFORE differencing, so the next
observed price still receives its proper multi-day return. Deliberately invalid
non-positive observations are retained as NaN placeholders during differencing,
so the invalid date and the immediately following observed date remain
undefined and are not bridged.

The five native-return series are then inner-joined by return endpoint date.

Run:
    python scripts/26_native_return_panel_corrected_audit.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

YRAW = RAW / "prices_yahoo_raw.csv"
BOGFILE = RAW / "bog" / "bog_historical_interbank_fx_usdghs.csv"
CURRENT = PROC / "returns_real.csv"

OUT_RET = TAB / "returns_real_native_common_corrected_candidate.csv"
OUT_PX = TAB / "prices_real_native_common_corrected_candidate.csv"
OUT_DESC = TAB / "native_common_corrected_candidate_descriptives.csv"
OUT_CMP = TAB / "native_common_corrected_candidate_comparison.csv"

START = pd.Timestamp("2015-01-01")
COMMS = ["cocoa", "gold", "brent", "wti"]

for p in [YRAW, BOGFILE, CURRENT]:
    if not p.exists():
        raise FileNotFoundError(p)

TAB.mkdir(parents=True, exist_ok=True)

yraw = pd.read_csv(YRAW, index_col=0, parse_dates=True).sort_index()
current = pd.read_csv(CURRENT, index_col=0, parse_dates=True).sort_index()

bog_df = pd.read_csv(BOGFILE, parse_dates=["date"]).sort_values("date")
bog = (bog_df.drop_duplicates("date", keep="first")
             .set_index("date")["mid_rate"]
             .astype(float)
             .sort_index()
             .dropna())

END = bog.loc[START:].last_valid_index()

print("=== CORRECTED NATIVE-RETURN COMMON-DATE AUDIT ===")
print(f"Fixed start: {START.date()}")
print(f"Last actual BoG observation: {END.date()}")
print()

ret_parts = {}
px_parts = {}
invalid_rows = []

for c in COMMS:
    # Own observed source dates only. Remove ordinary cross-series NaNs.
    s_obs = yraw[c].loc[START:END].dropna().astype(float).copy()

    # Keep actual non-positive observations in the index but mask their value.
    bad = s_obs <= 0
    bad_dates = list(s_obs.index[bad])
    s_for_log = s_obs.mask(bad)

    # This yields NaN on the invalid date and on the following observed date.
    r = 100 * np.log(s_for_log).diff()

    ret_parts[c] = r.rename(c)
    px_parts[c] = s_for_log.rename(c)

    invalid_rows.append({
        "series": c,
        "observed_price_rows": int(len(s_obs)),
        "nonpositive_rows": int(bad.sum()),
        "native_return_rows": int(r.notna().sum()),
        "first_obs": s_obs.index.min().date(),
        "last_obs": s_obs.index.max().date(),
        "nonpositive_dates": ";".join(str(d.date()) for d in bad_dates),
    })

# BoG on its own observed dates.
b = bog.loc[START:END].copy()
bret = -100 * np.log(b).diff()
ret_parts["cedi"] = bret.rename("cedi")
px_parts["cedi"] = b.rename("cedi")

invalid_rows.append({
    "series": "cedi",
    "observed_price_rows": int(len(b)),
    "nonpositive_rows": int((b <= 0).sum()),
    "native_return_rows": int(bret.notna().sum()),
    "first_obs": b.index.min().date(),
    "last_obs": b.index.max().date(),
    "nonpositive_dates": "",
})

inventory = pd.DataFrame(invalid_rows)
print("=== SOURCE-SERIES RETURN INVENTORY ===")
print(inventory.to_string(index=False))
print()

native = pd.concat(ret_parts.values(), axis=1, join="inner").dropna()
native = native.loc[START:END]

# Endpoint prices where present, for provenance only.
native_px = pd.concat(px_parts.values(), axis=1, join="outer").reindex(native.index)

native.to_csv(OUT_RET)
native_px.to_csv(OUT_PX)

print(f"Corrected native common panel: {native.shape}")
print(f"Date range: {native.index.min().date()} to {native.index.max().date()}")
print()

rows = []
for c in native.columns:
    x = native[c]
    nz = x[x != 0]
    rows.append({
        "series": c,
        "n": len(x),
        "zero_n": int((x == 0).sum()),
        "zero_pct": 100 * float((x == 0).mean()),
        "nonzero_abs_le_0_05_pctpt_pct":
            100 * float((nz.abs() <= 0.05).mean()) if len(nz) else np.nan,
        "nonzero_abs_le_0_10_pctpt_pct":
            100 * float((nz.abs() <= 0.10).mean()) if len(nz) else np.nan,
        "nonzero_abs_le_0_50_pctpt_pct":
            100 * float((nz.abs() <= 0.50).mean()) if len(nz) else np.nan,
        "min_return_pct": float(x.min()),
        "max_return_pct": float(x.max()),
    })
desc = pd.DataFrame(rows)
desc.to_csv(OUT_DESC, index=False, float_format="%.12f")

print("=== CORRECTED NATIVE COMMON DESCRIPTIVES ===")
print(desc.to_string(index=False))
print()

idx = native.index.intersection(current.index)
cmp_rows = []
print("=== CORRECTED NATIVE VS CURRENT ON COMMON DATES ===")
print(f"Common dates: {len(idx)}")
for c in native.columns:
    a = native.loc[idx, c]
    b2 = current.loc[idx, c]
    m = a.notna() & b2.notna()
    a, b2 = a[m], b2[m]
    row = {
        "series": c,
        "n_common": len(a),
        "pearson": a.corr(b2),
        "spearman": a.corr(b2, method="spearman"),
        "max_abs_diff": float((a-b2).abs().max()) if len(a) else np.nan,
    }
    cmp_rows.append(row)
    print(
        f"{c:8s} n={row['n_common']:4d} "
        f"pearson={row['pearson']:.12f} "
        f"spearman={row['spearman']:.12f} "
        f"max|diff|={row['max_abs_diff']:.12g}"
    )
print()
pd.DataFrame(cmp_rows).to_csv(OUT_CMP, index=False, float_format="%.12f")

COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))
C24 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
def count_window(idx, lo, hi):
    return int(((idx >= lo) & (idx <= hi)).sum())

print("=== STRESS-WINDOW COUNTS ===")
print(f"COVID:   {count_window(native.index, *COVID)}")
print(f"2024:    {count_window(native.index, *C24)}")
print(f"Combined:{count_window(native.index, *COVID) + count_window(native.index, *C24)}")
print()

print("=== WTI APRIL 2020 ===")
for d in pd.to_datetime(["2020-04-17","2020-04-20","2020-04-21","2020-04-22"]):
    status = "present" if d in native.index else "absent"
    val = None if d not in native.index else float(native.loc[d, "wti"])
    print(d.date(), status, val)
print()

print("=== DATES CURRENT HAS BUT CORRECTED NATIVE DOES NOT ===")
diff_dates = current.index.difference(native.index)
print(f"count={len(diff_dates)}")
print([str(d.date()) for d in diff_dates[:100]])
if len(diff_dates) > 100:
    print(f"... plus {len(diff_dates)-100} more")
print()

print("=== AUDIT OUTPUTS ===")
for p in [OUT_RET, OUT_PX, OUT_DESC, OUT_CMP]:
    print(p.relative_to(ROOT))
print("=== CORRECTED NATIVE-RETURN AUDIT COMPLETE ===")
