"""Audit and compare daily alignment choices, plus monthly macro zero codes.

NON-DESTRUCTIVE. It does not overwrite canonical processed data.

Run:
    python scripts/24_alignment_macro_integrity_audit.py
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
CPIFILE = RAW / "bog" / "real_sector_inflation_only.csv"
EXPFILE = RAW / "bog" / "merchandise_exports_only.csv"

CURRENT_PX = PROC / "prices_real.csv"
CURRENT_RET = PROC / "returns_real.csv"
MACRO = PROC / "ghana_macro_real.csv"

START = pd.Timestamp("2015-01-01")
COMMS = ["cocoa", "gold", "brent", "wti"]

for p in [YRAW, BOGFILE, CURRENT_PX, CURRENT_RET]:
    if not p.exists():
        raise FileNotFoundError(p)

TAB.mkdir(parents=True, exist_ok=True)

yraw = pd.read_csv(YRAW, index_col=0, parse_dates=True).sort_index()
current_px = pd.read_csv(CURRENT_PX, index_col=0, parse_dates=True).sort_index()
current_ret = pd.read_csv(CURRENT_RET, index_col=0, parse_dates=True).sort_index()

bog = pd.read_csv(BOGFILE, parse_dates=["date"]).sort_values("date")
bog = (bog.drop_duplicates("date", keep="first")
          .set_index("date")["mid_rate"].astype(float)
          .rename("cedi"))

# Candidate A: bounded short-gap fill
comm = yraw[COMMS].copy().loc[START:]
invalid_nonpositive = comm <= 0
comm = comm.mask(invalid_nonpositive)

actual_last = {c: comm[c].last_valid_index() for c in COMMS}
actual_last["cedi"] = bog.loc[START:].last_valid_index()
common_end = min(actual_last.values())

actual_first = {c: comm[c].first_valid_index() for c in COMMS}
actual_first["cedi"] = bog.loc[START:].first_valid_index()
common_start = max(actual_first.values())

source = comm.join(bog, how="outer").loc[common_start:common_end]
bidx = pd.date_range(common_start, common_end, freq="B")
cand_px = source.reindex(bidx).ffill(limit=3)

# Never fill deliberately invalid non-positive commodity observations.
for c in COMMS:
    bad_dates = invalid_nonpositive.index[invalid_nonpositive[c]]
    bad_dates = bad_dates.intersection(cand_px.index)
    if len(bad_dates):
        cand_px.loc[bad_dates, c] = np.nan

cand_ret = 100 * np.log(cand_px).diff()
cand_ret["cedi"] = -cand_ret["cedi"]
cand_ret = cand_ret.dropna()
cand_px_used = cand_px.loc[cand_ret.index.min():cand_ret.index.max()]

# Candidate B: strict actual-source common dates only
strict = yraw[COMMS].join(bog, how="inner").loc[START:]
for c in COMMS:
    strict = strict[strict[c] > 0]
strict = strict[strict["cedi"] > 0].dropna()
strict_ret = 100 * np.log(strict).diff().dropna()
strict_ret["cedi"] = -strict_ret["cedi"]

def descriptives(label, r):
    out = []
    for c in r.columns:
        x = r[c].dropna()
        nz = x[x != 0]
        out.append({
            "panel": label,
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
            "median_abs_nonzero_pctpt":
                float(nz.abs().median()) if len(nz) else np.nan,
        })
    return out

desc = pd.DataFrame(
    descriptives("current", current_ret)
    + descriptives("bounded_ffill_candidate", cand_ret)
    + descriptives("strict_common_candidate", strict_ret)
)

print("=== ALIGNMENT / ENDPOINT AUDIT ===")
print("Actual last source dates:")
for k, v in actual_last.items():
    print(f"  {k:8s}: {v.date()}")
print(f"Common source-supported endpoint: {common_end.date()}")
print()

print("Panel dimensions:")
print(f"  current:           {current_ret.shape}, {current_ret.index.min().date()} to {current_ret.index.max().date()}")
print(f"  bounded candidate: {cand_ret.shape}, {cand_ret.index.min().date()} to {cand_ret.index.max().date()}")
print(f"  strict candidate:  {strict_ret.shape}, {strict_ret.index.min().date()} to {strict_ret.index.max().date()}")
print()

old_only = current_ret.index.difference(cand_ret.index)
new_only = cand_ret.index.difference(current_ret.index)
print(f"Dates in CURRENT but not bounded candidate ({len(old_only)}):")
print([str(d.date()) for d in old_only])
print(f"Dates in bounded candidate but not CURRENT ({len(new_only)}):")
print([str(d.date()) for d in new_only])
print()

cmp_rows = []
for label, r in [("bounded_ffill_candidate", cand_ret),
                 ("strict_common_candidate", strict_ret)]:
    idx = current_ret.index.intersection(r.index)
    for c in current_ret.columns:
        a = current_ret.loc[idx, c]
        b = r.loc[idx, c]
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        cmp_rows.append({
            "candidate": label,
            "series": c,
            "n_common": len(a),
            "pearson": a.corr(b),
            "spearman": a.corr(b, method="spearman"),
            "max_abs_diff": float((a-b).abs().max()) if len(a) else np.nan,
        })
cmp = pd.DataFrame(cmp_rows)

print("=== OVERLAP COMPARISON TO CURRENT ===")
print(cmp.to_string(index=False))
print()

COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))
C24 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
def nwin(idx, win):
    return int(((idx >= win[0]) & (idx <= win[1])).sum())

print("=== STRESS-WINDOW SAMPLE COUNTS ===")
for label, r in [
    ("current", current_ret),
    ("bounded", cand_ret),
    ("strict", strict_ret),
]:
    print(f"{label:8s} COVID={nwin(r.index,COVID):3d}  2024={nwin(r.index,C24):3d}")
print()

print("=== CEDI DISTRIBUTION DESCRIPTIVES BY ALIGNMENT ===")
print(desc[desc["series"]=="cedi"].to_string(index=False))
print()

cand_px_used.to_csv(TAB / "prices_real_bounded_candidate.csv")
cand_ret.to_csv(TAB / "returns_real_bounded_candidate.csv")
strict.to_csv(TAB / "prices_real_strict_common_candidate.csv")
strict_ret.to_csv(TAB / "returns_real_strict_common_candidate.csv")
desc.to_csv(TAB / "alignment_candidate_descriptives.csv", index=False, float_format="%.12f")
cmp.to_csv(TAB / "alignment_candidate_comparison.csv", index=False, float_format="%.12f")

print("=== MONTHLY MACRO ZERO-CODE AUDIT ===")
if MACRO.exists():
    macro = pd.read_csv(MACRO, index_col=0, parse_dates=True).sort_index()
    for c in macro.columns:
        z = macro[c]
        zero_dates = list(z.index[z == 0])
        print(f"{c}: exact-zero dates = {[str(d.date()) for d in zero_dates]}")
        for d in zero_dates:
            loc = macro.index.get_loc(d)
            lo = max(0, loc-2)
            hi = min(len(macro), loc+3)
            print(macro.iloc[lo:hi][[c]].to_string())
            print()
else:
    print("Processed macro panel not found.")

if CPIFILE.exists():
    cpi_raw = pd.read_csv(CPIFILE, parse_dates=["date"]).sort_values("date")
    head = cpi_raw[cpi_raw["variable"].str.contains("Headline", case=False, na=False)].copy()
    zeros = head[head["value"] == 0]
    print("Raw BoG headline-inflation rows with value == 0:")
    print(zeros.to_string(index=False) if len(zeros) else "none")
    print()

if EXPFILE.exists():
    exp_raw = pd.read_csv(EXPFILE, parse_dates=["date"]).sort_values("date")
    total = exp_raw[exp_raw["variable"].str.contains(
        r"Merchandise Exports \(f\.o\.b\)", case=False, regex=True, na=False)].copy()
    zeros = total[total["value"] == 0]
    print("Raw BoG total-export rows with value == 0:")
    print(zeros.to_string(index=False) if len(zeros) else "none")
    print()

print("=== AUDIT OUTPUTS ===")
for name in [
    "prices_real_bounded_candidate.csv",
    "returns_real_bounded_candidate.csv",
    "prices_real_strict_common_candidate.csv",
    "returns_real_strict_common_candidate.csv",
    "alignment_candidate_descriptives.csv",
    "alignment_candidate_comparison.csv",
]:
    print(f"outputs/tables/{name}")

print("=== ALIGNMENT / MACRO INTEGRITY AUDIT COMPLETE ===")
