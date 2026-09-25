"""Build and audit a native-return common-date candidate panel.

This script is NON-DESTRUCTIVE. It does not overwrite data/processed files.

Design fixed before inspecting downstream statistical results:
1. Compute each asset's return on its own observed source dates.
2. Do not forward-fill prices to manufacture returns on source-missing dates.
3. Treat non-positive prices as invalid for log returns. The return on the
   invalid date and the immediately following source observation remain missing.
4. Sign-flip the BoG USD/GHS return so lower tail = cedi depreciation.
5. Inner-join the five return series by return endpoint date.
6. End no later than the last actual BoG observation.

Outputs are audit candidates only.

Run:
    python scripts/25_native_return_panel_audit.py
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
BOUNDED = TAB / "returns_real_bounded_candidate.csv"

OUT_RET = TAB / "returns_real_native_common_candidate.csv"
OUT_PX = TAB / "prices_real_native_common_candidate.csv"
OUT_DESC = TAB / "native_common_candidate_descriptives.csv"
OUT_CEDI_MONTH = TAB / "bog_cedi_monthly_source_coverage_audit.csv"

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
             .sort_index())

end = bog.loc[START:].last_valid_index()
print("=== NATIVE-RETURN COMMON-DATE PANEL AUDIT ===")
print(f"Fixed start: {START.date()}")
print(f"Last actual BoG observation: {end.date()}")
print()

# Native commodity returns: diff on the source index itself.
ret_parts = {}
px_parts = {}
for c in COMMS:
    s = yraw[c].loc[START:end].astype(float).copy()
    nonpos = s <= 0
    s_valid = s.mask(nonpos)

    # Because the invalid date remains in the indexed series as NaN, pandas
    # diff leaves both that date and the immediately following source row
    # undefined. This deliberately avoids bridging a log return across a
    # sign-changing price path such as WTI in April 2020.
    r = 100 * np.log(s_valid).diff()

    ret_parts[c] = r.rename(c)
    px_parts[c] = s_valid.rename(c)

    print(f"{c:8s}: source rows={s.notna().sum():4d}, "
          f"nonpositive={int(nonpos.fillna(False).sum()):2d}, "
          f"native returns={r.notna().sum():4d}")

# Native BoG return
b = bog.loc[START:end].copy()
bret = -100 * np.log(b).diff()
ret_parts["cedi"] = bret.rename("cedi")
px_parts["cedi"] = b.rename("cedi")
print(f"{'cedi':8s}: source rows={b.notna().sum():4d}, "
      f"native returns={bret.notna().sum():4d}")
print()

# Common endpoint dates across all five native return series.
native = pd.concat(ret_parts.values(), axis=1, join="inner").dropna()
native = native.loc[START:end]

# Endpoint prices for figures/provenance only.
native_px = pd.concat(px_parts.values(), axis=1, join="outer").reindex(native.index)

native.to_csv(OUT_RET)
native_px.to_csv(OUT_PX)

print(f"Native common panel: {native.shape}")
print(f"Date range: {native.index.min().date()} to {native.index.max().date()}")
print()

# Descriptives
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

print("=== NATIVE COMMON DESCRIPTIVES ===")
print(desc.to_string(index=False))
print()

# Compare to existing panel only on common dates.
def compare(label: str, other: pd.DataFrame):
    idx = native.index.intersection(other.index)
    print(f"=== NATIVE VS {label.upper()} ON COMMON DATES ===")
    print(f"Common dates: {len(idx)}")
    for c in native.columns:
        a = native.loc[idx, c]
        b2 = other.loc[idx, c]
        m = a.notna() & b2.notna()
        a, b2 = a[m], b2[m]
        print(
            f"{c:8s} n={len(a):4d} "
            f"pearson={a.corr(b2):.12f} "
            f"spearman={a.corr(b2, method='spearman'):.12f} "
            f"max|diff|={(a-b2).abs().max():.12g}"
        )
    print()

compare("current", current)
if BOUNDED.exists():
    bounded = pd.read_csv(BOUNDED, index_col=0, parse_dates=True).sort_index()
    compare("bounded candidate", bounded)

# Stress-window counts
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))
C24 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
def nwin(idx, lo, hi):
    return int(((idx >= lo) & (idx <= hi)).sum())

print("=== STRESS-WINDOW COUNTS ===")
print(f"COVID: {nwin(native.index, *COVID)}")
print(f"2024:  {nwin(native.index, *C24)}")
print(f"Combined: {nwin(native.index, *COVID) + nwin(native.index, *C24)}")
print()

# Explicit WTI dates
print("=== WTI APRIL 2020 IN NATIVE PANEL ===")
for d in pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21", "2020-04-22"]):
    print(d.date(), "present" if d in native.index else "absent",
          None if d not in native.index else float(native.loc[d, "wti"]))
print()

# BoG monthly source coverage and the resample-sum trap.
daily_lr = 100 * np.log(bog).diff()
monthly = pd.DataFrame({
    "source_obs": bog.resample("ME").count(),
    "return_obs": daily_lr.resample("ME").count(),
    "default_sum": daily_lr.resample("ME").sum(),
    "sum_min_count_1": daily_lr.resample("ME").sum(min_count=1),
    "first_level": bog.resample("ME").first(),
    "last_level": bog.resample("ME").last(),
})
monthly.to_csv(OUT_CEDI_MONTH, float_format="%.12f")

print("=== BOG MONTHLY SOURCE COVERAGE: 2025-01 TO 2025-05 ===")
print(monthly.loc["2025-01-01":"2025-05-31"].to_string())
print()
print("If default_sum is 0.0 while source_obs/return_obs are 0, the zero is an")
print("aggregation artifact caused by pandas sum() on an empty monthly bin.")
print()

print("=== AUDIT OUTPUTS ===")
for p in [OUT_RET, OUT_PX, OUT_DESC, OUT_CEDI_MONTH]:
    print(p.relative_to(ROOT))
print("=== NATIVE-RETURN PANEL AUDIT COMPLETE ===")
