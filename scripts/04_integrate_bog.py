"""Stage 0 — Integrate Bank of Ghana data as the primary cedi source.

Inputs (data/raw/):
    prices_yahoo_raw.csv                 uncleaned Yahoo panel (commodities + GHS=X)
    bog/bog_historical_interbank_fx_usdghs.csv   BoG interbank USD/GHS (1996->)
    bog/real_sector_inflation_only.csv           BoG monthly headline CPI YoY
    bog/merchandise_exports_only.csv             BoG monthly exports (USD m)

Outputs:
    data/processed/prices_real.csv, returns_real.csv   canonical analysis panel
        (cocoa, gold, brent, wti from Yahoo; cedi from BoG mid-rate;
         cedi returns sign-flipped: lower tail = depreciation)
    data/processed/ghana_macro_real.csv                monthly macro panel
    outputs/tables/cedi_crosscheck.csv                 BoG vs Yahoo comparison
    outputs/tables/cedi_crisis_resolution.csv          masked-date adjudication
    DECISIONS.md                                       appended decision log

Decisions implemented here (all printed):
    D1  BoG interbank mid-rate replaces Yahoo GHS=X as the cedi series.
    D2  BoG data is official: NO automatic spike-masking is applied to it;
        any |return|>15% day is REPORTED and adjudicated, not deleted.
    D3  Yahoo cedi is retained only as a cross-check column, never merged
        into the analysis panel.
    D4  Commodities keep the v4 cleaning rules (mask non-positive prices,
        business-day calendar, forward-fill limit 3 days).
    D5  The three previously masked crisis dates are resolved against BoG
        evidence and the resolution is written to a table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, TAB = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "outputs" / "tables"
DEC = ROOT / "DECISIONS.md"
CRISIS_DATES = ["2020-03-31", "2020-04-01", "2022-12-16"]
START = "2015-01-01"


def log_decision(text: str) -> None:
    print(f"DECISION: {text}")
    with DEC.open("a") as f:
        f.write(f"- {text}\n")


def main() -> None:
    DEC.touch()
    with DEC.open("a") as f:
        f.write(f"\n## Stage 0 — BoG integration ({pd.Timestamp.now():%Y-%m-%d})\n")

    # ---------------------------------------------------------- commodities
    yraw = pd.read_csv(RAW / "prices_yahoo_raw.csv", index_col=0, parse_dates=True)
    comm = yraw[["cocoa", "gold", "brent", "wti"]].copy()
    nonpos = (comm <= 0)
    if nonpos.any().any():
        for c in comm.columns[nonpos.any()]:
            log_decision(f"Masked non-positive {c} price(s) on "
                         f"{list(comm.index[nonpos[c]].date)} (log return undefined).")
        comm = comm.where(~nonpos)
    yahoo_ghs = yraw["ghs_usd"].rename("cedi_yahoo")

    # ---------------------------------------------------------- BoG cedi
    bog = pd.read_csv(RAW / "bog" / "bog_historical_interbank_fx_usdghs.csv",
                      parse_dates=["date"])
    bog = (bog.set_index("date").sort_index().loc[START:, "mid_rate"]
           .rename("cedi"))
    dup = bog.index.duplicated()
    if dup.any():
        log_decision(f"BoG file contained {int(dup.sum())} duplicate dates; "
                     "kept first occurrence.")
        bog = bog[~dup]
    log_decision("D1: BoG interbank USD/GHS mid-rate is the primary cedi "
                 f"series ({bog.index.min().date()} → {bog.index.max().date()}, "
                 f"{bog.size} obs); Yahoo GHS=X demoted to cross-check only (D3).")

    big = np.log(bog).diff().abs() > 0.15
    if big.any():
        log_decision(f"D2: BoG cedi days with |return|>15% reported, NOT masked: "
                     f"{list(bog.index[big].date)} — official data; adjudicated "
                     "in cedi_crisis_resolution.csv.")
    else:
        log_decision("D2: no BoG cedi day exceeds |15%|/day; no masking applied "
                     "to official data.")

    # ---------------------------------------------------------- crisis dates
    rows = []
    b_ret_all = np.log(bog).diff()
    y_ret_all = np.log(yahoo_ghs.where(yahoo_ghs > 0)).diff()
    for d in CRISIS_DATES:
        ts = pd.Timestamp(d)
        # adjudicate on a +/-3 business-day window: Yahoo often catches a
        # real move late as one artificial jump, so the exact date can differ
        win = b_ret_all.loc[ts - pd.Timedelta("5D"): ts + pd.Timedelta("5D")]
        max_bog = win.abs().max()
        cum_bog = win.sum()
        verdict = ("no comparable BoG move within +/-3bd -> Yahoo artifact; "
                   "BoG governs")
        if pd.notna(max_bog) and max_bog > 0.05:
            direction = "appreciation" if cum_bog < 0 else "depreciation"
            verdict = (f"GENUINE move confirmed by BoG within +/-3bd "
                       f"(max daily {max_bog:.1%}, cumulative {cum_bog:.1%} "
                       f"= {direction}); Yahoo mistimed it -> BoG series "
                       f"retains the move on its true dates")
        rows.append({"date": d,
                     "bog_log_ret_on_date": b_ret_all.get(ts, np.nan),
                     "bog_max_abs_ret_pm3bd": max_bog,
                     "yahoo_log_ret_on_date": y_ret_all.get(ts, np.nan),
                     "bog_level": bog.get(ts, np.nan),
                     "yahoo_level": yahoo_ghs.get(ts, np.nan),
                     "resolution": verdict})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "cedi_crisis_resolution.csv", index=False)
    log_decision("D5: masked-date adjudication written to "
                 "outputs/tables/cedi_crisis_resolution.csv: " +
                 "; ".join(f"{r.date}: {r.resolution.split('->')[1].strip()}"
                           for r in res.itertuples()))

    # ---------------------------------------------------------- cross-check
    both = pd.concat([bog, yahoo_ghs], axis=1, sort=True).dropna()
    ratio = both["cedi_yahoo"] / both["cedi"]
    garbage = both[(ratio > 2) | (ratio < 0.5)]
    if len(garbage):
        log_decision("Yahoo GHS=X contains outright garbage prints (level off "
                     "by >2x vs BoG): "
                     + "; ".join(f"{d.date()} yahoo={r.cedi_yahoo:.2f} vs "
                                 f"bog={r.cedi:.2f}"
                                 for d, r in garbage.iterrows())
                     + " — these alone destroy naive level/return "
                       "correlations and are the strongest argument for D1.")
    clean = both[(ratio <= 2) & (ratio >= 0.5)]
    stats = {}
    for tag, frame in [("raw", both), ("excl_garbage", clean)]:
        r = np.log(frame).diff().dropna()
        gap = (frame["cedi_yahoo"] / frame["cedi"] - 1).abs()
        stats[tag] = {
            "n_common_days": len(frame),
            "level_correlation": frame["cedi"].corr(frame["cedi_yahoo"]),
            "return_correlation": r["cedi"].corr(r["cedi_yahoo"]),
            "n_days_gap_gt_2pct": int((gap > 0.02).sum()),
            "median_gap_pct": float(gap.median() * 100),
            "max_gap_pct": float(gap.max() * 100),
        }
    cc = pd.DataFrame(stats).T
    cc.round(4).to_csv(TAB / "cedi_crosscheck.csv")
    gap_c = (clean["cedi_yahoo"] / clean["cedi"] - 1).abs()
    clean[gap_c > 0.02].assign(gap_pct=(gap_c[gap_c > 0.02] * 100).round(2)) \
        .to_csv(TAB / "cedi_crosscheck_disagreements.csv")
    print("Cross-check (raw | excl. garbage prints):")
    print(cc.round(3).to_string())

    # ---------------------------------------------------------- final panel
    panel = comm.join(bog, how="outer").loc[START:]
    panel = panel.asfreq("B").ffill(limit=3).dropna()
    rets = 100 * np.log(panel).diff().dropna()
    rets["cedi"] = -rets["cedi"]  # ORIENTATION: lower tail = depreciation
    log_decision("Orientation re-verified: cedi returns sign-flipped so the "
                 "lower tail = depreciation = bad-for-Ghana for all five "
                 "series; commodity lower tails are price crashes.")
    panel = panel.loc[rets.index.min():]
    panel.to_csv(PROC / "prices_real.csv")
    rets.to_csv(PROC / "returns_real.csv")
    print(f"Canonical panel: {rets.shape[0]} obs x {rets.shape[1]} "
          f"({rets.index.min().date()} → {rets.index.max().date()})")

    # ---------------------------------------------------------- monthly macro
    infl = pd.read_csv(RAW / "bog" / "real_sector_inflation_only.csv",
                       parse_dates=["date"])
    infl = (infl[infl["variable"].str.contains("Headline", case=False)]
            .set_index("date")["value"].sort_index().rename("cpi_yoy"))
    exp_ = pd.read_csv(RAW / "bog" / "merchandise_exports_only.csv",
                       parse_dates=["date"])
    exp_all_vars = sorted(exp_["variable"].unique())
    exp_ = exp_[exp_["variable"].str.contains(
        r"Merchandise Exports \(f\.o\.b\)", case=False, regex=True)]
    exp_ = (exp_.set_index("date")["value"].sort_index()
            .rename("exports_usd_m"))
    n_neg = int((exp_ < 0).sum())
    log_decision(
        "BUGFIX: merchandise_exports_only.csv stacks 8 variables long-form "
        f"(found: {exp_all_vars}); the initial extraction took the 'value' "
        "column unfiltered, which mixed in 'Merchandise Exports Less "
        "Imports_Trade Balance' (legitimately negative) and commodity "
        "sub-components. Filtered to 'Merchandise Exports (f.o.b)' only "
        f"({exp_.notna().sum()} obs). Remaining negative values after "
        f"filter: {n_neg} (should be 0 for a pure f.o.b. level).")
    if n_neg:
        log_decision(f"WARNING: {n_neg} negative f.o.b. export value(s) "
                     f"remain after filtering: "
                     f"{exp_[exp_<0].round(2).to_dict()} — reported, not "
                     "altered; investigate against the BoG source table "
                     "before using in Stage 6.")

    # The BoG source table encodes a reporting gap (2023-05 to 2023-12) as
    # a literal 0.00 rather than blank — exports do not fall to zero.
    # Treat exact zeros as missing (not a real observation); log(0) would
    # otherwise silently produce -inf in the Stage 6 growth regression.
    zero_months = exp_[exp_ == 0.0]
    if len(zero_months):
        log_decision(
            f"BoG export table encodes a reporting gap as literal 0.00 for "
            f"{len(zero_months)} months ({zero_months.index.min():%Y-%m} to "
            f"{zero_months.index.max():%Y-%m}) — exports did not actually "
            f"collapse to zero; these are missing data, not observations. "
            f"Masked to NaN before use.")
        exp_ = exp_.mask(exp_ == 0.0)
    cedi_m = (100 * np.log(bog).diff().resample("ME").sum()
              .rename("cedi_depreciation_pct_m"))  # +ve = depreciation (raw BoG)
    macro = pd.concat([infl.resample("ME").last(), exp_.resample("ME").last(),
                       cedi_m], axis=1).loc[START:]
    macro.to_csv(PROC / "ghana_macro_real.csv")
    log_decision(f"Monthly macro panel (BoG): CPI YoY "
                 f"{infl.dropna().index.min():%Y-%m}→{infl.dropna().index.max():%Y-%m}, "
                 f"exports {exp_.dropna().index.min():%Y-%m}→"
                 f"{exp_.dropna().index.max():%Y-%m}; cedi monthly depreciation "
                 "kept in raw orientation (+ = depreciation) for Stage 6.")
    print("Stage 0 complete.")


if __name__ == "__main__":
    sys.exit(main())
