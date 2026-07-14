"""Stage 0 — Build the real analysis panel with Bank of Ghana cedi primary.

Inputs (data/raw/):
  yahoo/prices_yahoo_raw.csv            uncleaned Yahoo download (5 series)
  bog/bog_historical_interbank_fx_usdghs.csv   BoG daily interbank USD/GHS
  bog/real_sector_inflation_only.csv    BoG monthly headline CPI y/y
  bog/merchandise_exports_only.csv      BoG monthly merchandise exports

Outputs:
  data/processed/prices_real.csv, returns_real.csv   (cedi = BoG, oriented)
  data/processed/ghana_macro_real.csv                (monthly macro)
  outputs/tables/bog_yahoo_crosscheck.csv
  outputs/tables/crisis_date_resolution.csv

Decisions implemented here (also recorded in DECISIONS.md):
  D1 Cedi primary = BoG interbank mid-rate; Yahoo GHS=X demoted to
     cross-check only (known stale quotes/spikes).
  D2 Commodity cleaning identical to the audited fetcher: mask
     non-positive prices (negative WTI 2020-04-20), business-day
     calendar, forward-fill limited to 3 days, no other imputation.
  D3 Orientation: cedi returns are sign-flipped (lower tail =
     depreciation = bad-for-Ghana), column named 'cedi'; the price
     column keeps the GHS-per-USD level for interpretability.
  D4 Crisis dates masked on Yahoo (2020-03-31/04-01, 2022-12-16) are
     adjudicated with BoG: if BoG shows a comparable move, the move is
     real and stays (it is in the BoG series we now use); the Yahoo
     masking episode is documented, not applied to BoG.
  D5 No bad-tick filter is applied to the BoG series itself: it is the
     official reference series; any large move in it is treated as
     real. (Its own quality is assessed by the cross-check table.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

START = "2015-01-01"


def main() -> None:
    # ---------------------------------------------------------- commodities
    yah = pd.read_csv(RAW / "yahoo" / "prices_yahoo_raw.csv",
                      index_col=0, parse_dates=True).sort_index()
    comm = yah[["cocoa", "gold", "brent", "wti"]].loc[START:]
    nonpos = comm <= 0
    if nonpos.any().any():
        for c in comm.columns[nonpos.any()]:
            days = list(comm.index[nonpos[c]].date)
            print(f"[D2] Masked non-positive {c} price(s): {days}")
        comm = comm.where(~nonpos)

    # ------------------------------------------------------------- BoG cedi
    bog = pd.read_csv(RAW / "bog" / "bog_historical_interbank_fx_usdghs.csv",
                      parse_dates=["date"])
    bog_mid = (bog.set_index("date")["mid_rate"].sort_index().loc[START:]
               .rename("cedi"))
    dup = bog_mid.index.duplicated().sum()
    if dup:
        print(f"[D1] Dropped {dup} duplicate BoG dates (kept first).")
        bog_mid = bog_mid[~bog_mid.index.duplicated()]
    print(f"[D1] BoG interbank mid-rate: {bog_mid.size} obs, "
          f"{bog_mid.index.min().date()} → {bog_mid.index.max().date()}")

    # ----------------------------------------------------- Yahoo cross-check
    y_ghs = yah["ghs_usd"].loc[START:]
    both = pd.concat([bog_mid, y_ghs.rename("yahoo")], axis=1, sort=True).dropna()
    lvl_corr = both["cedi"].corr(both["yahoo"])
    ret_b = np.log(both["cedi"]).diff()
    ret_y = np.log(both["yahoo"]).diff()
    ret_corr = ret_b.corr(ret_y)
    gap = (both["yahoo"] / both["cedi"] - 1).abs()
    disagree = both[gap > 0.02].assign(gap_pct=(gap[gap > 0.02] * 100).round(2))
    disagree.to_csv(TAB / "bog_yahoo_crosscheck_disagreements.csv")
    pd.DataFrame([{
        "overlap_days": len(both), "level_corr": lvl_corr,
        "return_corr": ret_corr, "n_days_gap_gt_2pct": len(disagree),
        "worst_gap_pct": float(gap.max() * 100),
        "worst_gap_date": str(gap.idxmax().date()),
    }]).round(4).to_csv(TAB / "bog_yahoo_crosscheck.csv", index=False)
    print(f"[D1] Cross-check: {len(both)} overlap days | level corr "
          f"{lvl_corr:.4f} | return corr {ret_corr:.3f} | "
          f"{len(disagree)} days with >2% level gap (worst "
          f"{gap.max()*100:.1f}% on {gap.idxmax().date()})")

    # -------------------------------------------- crisis date adjudication
    rows = []
    for d in ["2020-03-31", "2020-04-01", "2022-12-16", "2016-05-13",
              "2016-05-16"]:
        ts = pd.Timestamp(d)
        w = bog_mid.loc[ts - pd.Timedelta("7D"): ts + pd.Timedelta("7D")]
        r = np.log(w).diff() * 100
        move = r.get(ts, np.nan)
        rows.append({"date": d,
                     "bog_daily_move_pct": None if pd.isna(move)
                     else round(float(move), 3),
                     "bog_window_max_abs_move_pct":
                         round(float(r.abs().max()), 3) if len(r) else None,
                     "verdict": ("real move retained (BoG primary)"
                                 if not pd.isna(move) and abs(move) > 1.0
                                 else "no comparable BoG move — Yahoo tick "
                                      "was an artifact; BoG series used")})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "crisis_date_resolution.csv", index=False)
    print("[D4] Crisis-date adjudication:")
    print(res.to_string(index=False))

    # -------------------------------------------------- assemble the panel
    panel = pd.concat([comm, bog_mid], axis=1).sort_index()
    panel = panel.asfreq("B").ffill(limit=3)
    n_incomplete = int(panel.isna().any(axis=1).sum())
    panel = panel.dropna()
    print(f"[D2] Business-day panel: dropped {n_incomplete} incomplete rows "
          f"(holiday-calendar mismatches beyond the 3-day fill limit)")

    rets = 100 * np.log(panel).diff().dropna()
    rets["cedi"] = -rets["cedi"]                       # D3 orientation
    print("[D3] Cedi returns sign-flipped: lower tail = depreciation.")
    panel = panel.loc[rets.index.min():]

    panel.to_csv(PROC / "prices_real.csv")
    rets.to_csv(PROC / "returns_real.csv")
    print(f"PANEL: {rets.shape[0]} obs x {rets.shape[1]} series, "
          f"{rets.index.min().date()} → {rets.index.max().date()}")

    # --------------------------------------------------- monthly macro (S6)
    infl = pd.read_csv(RAW / "bog" / "real_sector_inflation_only.csv",
                       parse_dates=["date"])
    infl = (infl[infl["variable"].str.contains("Headline")]
            .groupby("date")["value"].first().sort_index().rename("cpi_yoy"))
    exp_ = pd.read_csv(RAW / "bog" / "merchandise_exports_only.csv",
                       parse_dates=["date"])
    exp_ = (exp_.groupby("date")["value"].first().sort_index()
            .rename("exports_usd_m"))
    cedi_m = np.log(bog_mid.resample("ME").last()).diff().mul(100) \
        .rename("cedi_depreciation_pct_m")
    # The BoG scrape mixes month-start rows with annual (31 Dec) rows;
    # aggregate to month periods (first value) to get one row per month.
    infl = infl.groupby(infl.index.to_period("M")).first()
    infl.index = infl.index.to_timestamp("M")
    exp_ = exp_.groupby(exp_.index.to_period("M")).first()
    exp_.index = exp_.index.to_timestamp("M")
    macro = pd.concat([infl, exp_, cedi_m], axis=1, sort=True).loc[START:]
    macro.to_csv(PROC / "ghana_macro_real.csv")
    print(f"MACRO: {macro.shape[0]} months | inflation "
          f"{infl.dropna().index.max().date()} | exports "
          f"{exp_.dropna().index.max().date()}")


if __name__ == "__main__":
    sys.exit(main())
