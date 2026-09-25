"""Fetch REAL market and macro data. Run this LOCALLY (needs internet).

    pip install yfinance wbgapi
    python scripts/01_fetch_real_data.py

Writes data/processed/prices_yahoo_clean.csv and returns_yahoo_clean.csv.
These are NOT the canonical analysis inputs -- scripts/04_integrate_bog.py
reads data/raw/prices_yahoo_raw.csv (written here, unchanged) and produces
the actual canonical prices_real.csv / returns_real.csv with the BoG cedi
rate substituted in. Previously this script wrote directly to
prices_real.csv / returns_real.csv, the same filenames script 04 also
writes -- meaning a rerun of this script after script 04 would silently
overwrite the validated BoG-based canonical panel with the Yahoo-proxy
version, no error or warning. Run this first, then:

    python scripts/04_integrate_bog.py
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

Sample window
-------------
START and END define the inclusive publication sample window. The Yahoo
Finance API uses an exclusive end boundary, so the download request is
advanced by one calendar day and the resulting panel is then explicitly
sliced back to START:END.

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

START = "2015-01-01"
END = "2026-07-15"

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

    # 4. ORIENTATION: GHS=X quotes cedis per USD, so a *positive* return is
    # cedi depreciation (bad for Ghana), while for the commodities a
    # *negative* return is the bad day. Tail-dependence analysis needs all
    # series oriented the same way: flip the cedi's return sign and rename
    # the series 'cedi', so that the LOWER tail = bad-for-Ghana day for
    # every column. (Equivalent to using USD-per-GHS returns.)
    if "ghs_usd" in rets:
        rets["ghs_usd"] = -rets["ghs_usd"]
        rets = rets.rename(columns={"ghs_usd": "cedi"})
        px = px.rename(columns={"ghs_usd": "cedi"})  # price stays GHS/USD level
        print("Oriented cedi series: returns sign-flipped so lower tail = "
              "depreciation; column renamed ghs_usd -> cedi "
              "(price column still holds the GHS-per-USD level).")

    px = px.loc[rets.index.min():]
    return px, rets


def main(start: str = START, end: str = END) -> None:
    import yfinance as yf

    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    # yfinance's end= is exclusive (confirmed against its own documentation:
    # "for end='2023-01-01', the last data point will be '2022-12-31'"),
    # while START/END here are meant as an inclusive publication window.
    # Advance the request boundary by one day so the declared END date is
    # actually included in what gets requested.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frames = {}
    for name, tic in TICKERS.items():
        px = yf.download(tic, start=start, end=end_exclusive, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.DataFrame):  # newer yfinance returns 2-D
            px = px.iloc[:, 0]
        px.name = name
        frames[name] = px
        print(f"{name:8s} {tic:6s} {px.dropna().index.min().date()} → "
              f"{px.dropna().index.max().date()}  ({px.dropna().size} obs)")

    raw = pd.concat(frames.values(), axis=1, sort=True).dropna(how="all")
    # Second, independent safeguard: even if yfinance's boundary behavior
    # ever changes, the saved raw panel cannot extend beyond the declared
    # publication cutoff.
    raw = raw.loc[pd.Timestamp(start):pd.Timestamp(end)]
    raw.to_csv(OUT / "prices_yahoo_raw.csv")  # keep the uncleaned original;
    # this is the file scripts/04_integrate_bog.py actually reads

    prices, rets = clean_panel(raw)
    # NOT prices_real.csv / returns_real.csv -- those filenames are
    # reserved exclusively for scripts/04_integrate_bog.py's canonical,
    # BoG-based output. Writing there from this script would silently
    # overwrite the validated canonical panel on any rerun.
    prices.to_csv(PROC / "prices_yahoo_clean.csv")
    rets.to_csv(PROC / "returns_yahoo_clean.csv")
    print(f"Clean panel (Yahoo-only, NOT canonical): {rets.shape[0]} obs x "
          f"{rets.shape[1]} series, {rets.index.min().date()} → "
          f"{rets.index.max().date()}")

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