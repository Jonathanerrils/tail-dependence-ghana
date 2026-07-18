import sys
sys.path.insert(0, "src")
import openpyxl
import numpy as np
import pandas as pd

COLS = {"brent": 2, "wti": 4, "cocoa": 11, "gold": 69}

# ------------------------------------------------------------- Pink Sheet
wb = openpyxl.load_workbook("data/raw/CMO-Historical-Data-Monthly.xlsx",
                            read_only=True, data_only=True)
ws = wb["Monthly Prices"]
data = list(ws.iter_rows(min_row=7, values_only=True))
dates, vals = [], {k: [] for k in COLS}
for row in data:
    label = row[0]
    if not label or "M" not in str(label):
        continue
    y, m = str(label).split("M")
    dates.append(pd.Timestamp(int(y), int(m), 1))
    for name, idx in COLS.items():
        vals[name].append(row[idx])

pink = pd.DataFrame(vals, index=pd.DatetimeIndex(dates)).apply(pd.to_numeric, errors="coerce")
pink = pink.loc["2015-01-01":]
print("Pink Sheet monthly panel:", pink.shape, pink.index.min(), "->", pink.index.max())
print(pink.tail(3))

# ------------------------------------------------------------- Yahoo futures
px = pd.read_csv("data/processed/prices_real.csv", index_col=0, parse_dates=True)
# Two resampling conventions, compared explicitly: Pink Sheet's documented
# methodology is a MONTHLY AVERAGE of daily prices, not an end-of-month
# snapshot, so the mean-based series is the methodologically matched
# comparison; the snapshot version is retained to show how much of any
# gap is a sampling-convention artifact rather than a genuine roll effect.
yahoo_mean = px.resample("MS").mean()
yahoo_last = px.resample("MS").last()
for label, yahoo_m in [("mean", yahoo_mean), ("snapshot", yahoo_last)]:
    yahoo_m = yahoo_m.loc[pink.index.min():pink.index.max()]
    results = []
    for name in COLS:
        both = pd.concat([
            np.log(pink[name]).diff().rename("pink_ret"),
            np.log(yahoo_m[name]).diff().rename("yahoo_ret"),
        ], axis=1).dropna()
        corr = both["pink_ret"].corr(both["yahoo_ret"])
        diff = (both["yahoo_ret"] - both["pink_ret"])
        results.append({
            "series": name, "n_months": len(both),
            "return_correlation": corr,
            "mean_abs_monthly_diff_pct": float(diff.abs().mean()),
            "max_abs_monthly_diff_pct": float(diff.abs().max()),
            "max_diff_date": str(diff.abs().idxmax().date()),
        })
        if label == "mean":
            both.to_csv(f"outputs/tables/pinksheet_vs_yahoo_{name}_returns.csv")
    res = pd.DataFrame(results)
    res.round(4).to_csv(f"outputs/tables/pinksheet_rolleffect_summary_{label}.csv", index=False)
    print(f"\n=== Pink Sheet vs Yahoo-futures monthly return comparison "
          f"(Yahoo resampled by {label}) ===")
    print(res.round(4).to_string(index=False))
