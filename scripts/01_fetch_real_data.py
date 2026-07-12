"""Fetch REAL market and macro data. Run this LOCALLY (needs internet).

    pip install yfinance wbgapi
    python scripts/01_fetch_real_data.py

Writes data/processed/prices_real.csv and returns_real.csv in the schema
the analysis pipeline expects, so afterwards run:

    python scripts/02_run_pipeline.py --data real
    python scripts/03_build_dashboard.py --data real

Sources
-------
* Futures/spot via Yahoo Finance:
    CC=F  ICE cocoa futures (USD/tonne)
    GC=F  COMEX gold futures
    BZ=F  Brent crude futures
    CL=F  WTI crude futures
    GHS=X USD/GHS exchange rate
* Ghana macro via World Bank (wbgapi): CPI inflation (FP.CPI.TOTL.ZG,
  annual) — replace/augment with Bank of Ghana monthly CPI and the
  interbank cedi rate from https://www.bog.gov.gh for the frequency the
  paper needs (Yahoo's GHS=X is a low-quality proxy).

Cleaning decisions (deliberate, logged to console)
--------------------------------------------------
* Non-positive prices are masked, not silently dropped. This matters:
  the sample includes 2020-04-20, when front-month WTI (CL=F) settled at
  -$37.63 — log returns are undefined across that day. The day is
  reported, masked, then bridged by the limited forward-fill below.
* GHS=X ticks moving >15% in a day are masked as bad quotes (Yahoo's
  cedi series contains stale quotes and spikes).
* Calendar: business-day index; forward-fill LIMITED to 3 days
  (holidays). Unlimited ffill can fabricate stale prices across long
  gaps and manufacture artificial zero-volatility stretches.
* GHS=X weekend/holiday quotes (it prints more days than the futures)
  are folded onto the business-day calendar by the alignment step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

TICKERS = {
    "cocoa": "CC=F",
    "gold": "GC=F",
    "brent": "BZ=F",
    "wti": "CL=F",
    "ghs_usd": "GHS=X",
}


def clean_panel(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the documented cleaning rules; return (prices, pct log returns)."""
    px = prices.copy()

    # 1. GHS bad-tick mask (before alignment, on the raw quote series)
    if "ghs_usd" in px:
        lr = np.log(px["ghs_usd"].where(px["ghs_usd"] > 0)).diff()
        bad = lr.abs() > 0.15
        if bad.any():
            days = list(px.index[bad].date)
            print(f"Masked {int(bad.sum())} suspect GHS/USD ticks (>15%/day): "
                  f"{days[:8]}{' …' if len(days) > 8 else ''}")
            px.loc[bad, "ghs_usd"] = np.nan

    # 2. Non-positive prices (e.g. negative WTI on 2020-04-20) — mask & report
    nonpos = (px <= 0)
    if nonpos.any().any():
        for c in px.columns[nonpos.any()]:
            days = list(px.index[nonpos[c]].date)
            print(f"Masked {len(days)} non-positive {c} price(s): {days} "
                  f"(log return undefined; day bridged by limited ffill)")
        px = px.where(~nonpos)

    # 3. Business-day calendar; forward-fill holidays only (max 3 days)
    px = px.asfreq("B").ffill(limit=3).dropna()

    rets = 100 * np.log(px).diff().dropna()
    px = px.loc[rets.index.min():]
    return px, rets


def main(start: str = "2015-01-01") -> None:
    import yfinance as yf

    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    frames = {}
    for name, tic in TICKERS.items():
        px = yf.download(tic, start=start, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):  # newer yfinance returns 2-D
            px = px.iloc[:, 0]
        px.name = name
        frames[name] = px
        print(f"{name:8s} {tic:6s} {px.dropna().index.min().date()} → "
              f"{px.dropna().index.max().date()}  ({px.dropna().size} obs)")

    raw = pd.concat(frames.values(), axis=1, sort=True).dropna(how="all")
    raw.to_csv(OUT / "prices_yahoo_raw.csv")  # keep the uncleaned original

    prices, rets = clean_panel(raw)
    prices.to_csv(PROC / "prices_real.csv")
    rets.to_csv(PROC / "returns_real.csv")
    print(f"Clean panel: {rets.shape[0]} obs x {rets.shape[1]} series, "
          f"{rets.index.min().date()} → {rets.index.max().date()}")

    # --- Ghana macro (annual CPI from World Bank; swap in BoG monthly data)
    try:
        import wbgapi as wb

        cpi = wb.data.DataFrame("FP.CPI.TOTL.ZG", "GHA").T
        cpi.index = cpi.index.str.replace("YR", "").astype(int)
        cpi.columns = ["cpi_inflation_yoy"]
        cpi.to_csv(OUT / "ghana_cpi_worldbank_annual.csv")
        print("World Bank Ghana CPI saved (annual).")
    except Exception as e:  # noqa: BLE001
        print(f"World Bank fetch skipped: {e}")

    print("\nNOTE: for monthly inflation, the cedi interbank rate and export "
          "receipts, download the Bank of Ghana time-series workbooks and "
          "save them under data/raw/ — see README §Data. Yahoo's GHS=X is a "
          "proxy; prefer BoG for the cedi before final results.")


if __name__ == "__main__":
    main()
