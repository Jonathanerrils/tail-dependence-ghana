"""Audit a synchronized common-observation return panel.

NON-DESTRUCTIVE. This candidate enforces the same return interval for all five
series by:

1. using only dates with an actual source observation for every series;
2. requiring all five price levels to be positive on the endpoint dates;
3. computing log returns only after the common price-date intersection, so each
   row uses the same previous common date for all five assets;
4. treating an endpoint return as invalid if any source had an actual
   non-positive price between the previous common date and that endpoint;
5. ending at the last actual BoG observation.

It does not overwrite canonical data.

Run:
    python scripts/27_synchronized_common_interval_audit.py
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
NATIVE = TAB / "returns_real_native_common_corrected_candidate.csv"

OUT_PX = TAB / "prices_real_synchronized_common_candidate.csv"
OUT_RET = TAB / "returns_real_synchronized_common_candidate.csv"
OUT_DESC = TAB / "synchronized_common_candidate_descriptives.csv"
OUT_CMP = TAB / "synchronized_common_candidate_comparison.csv"
OUT_INTERVAL = TAB / "synchronized_common_interval_lengths.csv"

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
             .set_index("date")["mid_rate"].astype(float).sort_index().dropna())

END = bog.loc[START:].last_valid_index()

px = yraw[COMMS].loc[START:END].join(bog.rename("cedi"), how="outer")

nonpositive = pd.DataFrame(False, index=px.index, columns=px.columns)
for c in COMMS:
    nonpositive[c] = px[c].notna() & (px[c] <= 0)
nonpositive["cedi"] = px["cedi"].notna() & (px["cedi"] <= 0)

print("=== SYNCHRONIZED COMMON-INTERVAL AUDIT ===")
print(f"Fixed start: {START.date()}")
print(f"Last actual BoG observation: {END.date()}")
print("Actual non-positive observations:")
for c in px.columns:
    ds = list(px.index[nonpositive[c]])
    print(f"  {c:8s}: {[str(d.date()) for d in ds]}")
print()

valid_common = px.notna().all(axis=1) & (px > 0).all(axis=1)
common_px = px.loc[valid_common].copy()

sync_ret = 100 * np.log(common_px).diff()
sync_ret["cedi"] = -sync_ret["cedi"]

invalid_endpoint_rows = set()
common_dates = common_px.index
for i in range(1, len(common_dates)):
    prev_d = common_dates[i-1]
    curr_d = common_dates[i]
    between = nonpositive.loc[
        (nonpositive.index > prev_d) & (nonpositive.index <= curr_d)
    ]
    if between.any().any():
        invalid_endpoint_rows.add(curr_d)

if invalid_endpoint_rows:
    sync_ret.loc[list(invalid_endpoint_rows), :] = np.nan

sync_ret = sync_ret.dropna()
sync_px_used = common_px.reindex(sync_ret.index)

sync_ret.to_csv(OUT_RET)
sync_px_used.to_csv(OUT_PX)

print(f"Common actual-source price dates: {len(common_px)}")
print("Invalid synchronized endpoint rows due to intervening non-positive price:",
      [str(d.date()) for d in sorted(invalid_endpoint_rows)])
print(f"Synchronized return panel: {sync_ret.shape}")
print(f"Date range: {sync_ret.index.min().date()} to {sync_ret.index.max().date()}")
print()

all_interval = pd.DataFrame(index=common_px.index[1:])
all_interval["prev_common_date"] = common_px.index[:-1].values
all_interval["calendar_days"] = (
    all_interval.index.to_series().values
    - pd.to_datetime(all_interval["prev_common_date"]).values
) / np.timedelta64(1, "D")
all_interval["kept_return"] = all_interval.index.isin(sync_ret.index)
all_interval.to_csv(OUT_INTERVAL, index_label="end_date")

print("=== SYNCHRONIZED INTERVAL LENGTHS ===")
kept = all_interval[all_interval["kept_return"]]
print(kept["calendar_days"].value_counts().sort_index().to_string())
print(f"median calendar interval: {kept['calendar_days'].median():.1f} days")
print(f"max calendar interval: {kept['calendar_days'].max():.0f} days")
print()

rows = []
for c in sync_ret.columns:
    x = sync_ret[c]
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

print("=== SYNCHRONIZED PANEL DESCRIPTIVES ===")
print(desc.to_string(index=False))
print()

comparisons = []
for label, other_path in [("current", CURRENT), ("native_common_corrected", NATIVE)]:
    if not other_path.exists():
        continue
    other = pd.read_csv(other_path, index_col=0, parse_dates=True).sort_index()
    idx = sync_ret.index.intersection(other.index)
    print(f"=== SYNCHRONIZED VS {label.upper()} ===")
    print(f"Common endpoint dates: {len(idx)}")
    for c in sync_ret.columns:
        a = sync_ret.loc[idx, c]
        b = other.loc[idx, c]
        m = a.notna() & b.notna()
        a, b = a[m], b[m]
        row = {
            "comparison": label,
            "series": c,
            "n_common": len(a),
            "pearson": a.corr(b),
            "spearman": a.corr(b, method="spearman"),
            "max_abs_diff": float((a-b).abs().max()) if len(a) else np.nan,
        }
        comparisons.append(row)
        print(
            f"{c:8s} n={len(a):4d} "
            f"pearson={row['pearson']:.12f} "
            f"spearman={row['spearman']:.12f} "
            f"max|diff|={row['max_abs_diff']:.12g}"
        )
    print()

pd.DataFrame(comparisons).to_csv(OUT_CMP, index=False, float_format="%.12f")

COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))
C24 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

def nwin(idx, lo, hi):
    return int(((idx >= lo) & (idx <= hi)).sum())

print("=== STRESS-WINDOW COUNTS ===")
print(f"COVID:    {nwin(sync_ret.index, *COVID)}")
print(f"2024:     {nwin(sync_ret.index, *C24)}")
print(f"Combined: {nwin(sync_ret.index, *COVID) + nwin(sync_ret.index, *C24)}")
print()

print("=== WTI APRIL 2020 ===")
for d in pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21", "2020-04-22"]):
    status = "present" if d in sync_ret.index else "absent"
    val = None if d not in sync_ret.index else float(sync_ret.loc[d, "wti"])
    print(d.date(), status, val)
print()

print("=== AUDIT OUTPUTS ===")
for p in [OUT_PX, OUT_RET, OUT_DESC, OUT_CMP, OUT_INTERVAL]:
    print(p.relative_to(ROOT))
print("=== SYNCHRONIZED COMMON-INTERVAL AUDIT COMPLETE ===")
