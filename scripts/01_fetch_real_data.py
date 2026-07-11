"""Fetch REAL market and macro data. Run this LOCALLY (needs internet).

    pip install yfinance wbgapi
    python scripts/01_fetch_real_data.py

Outputs the same file schema the pipeline expects, so after running this
you can point scripts/02_run_pipeline.py at the real files by passing
--data real.

Sources
-------
* Futures/spot via Yahoo Finance:
    CC=F  ICE cocoa futures (USD/tonne)
    GC=F  COMEX gold futures
    BZ=F  Brent crude futures
    CL=F  WTI crude futures
    GHS=X USD/GHS exchange rate
* Ghana macro via World Bank (wbgapi): CPI inflation (FP.CPI.TOTL.ZG,
  annual) — replace/augment with Bank of Ghana monthly CPI and export
  receipts from https://www.bog.gov.gh (Statistics > Time Series Data)
  for the frequency the paper needs.
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


def main(start: str = "2015-01-01") -> None:
    import yfinance as yf

    frames = {}
    for name, tic in TICKERS.items():
        px = yf.download(tic, start=start, auto_adjust=True, progress=False)["Close"]
        px.name = name
        frames[name] = px
        print(f"{name:8s} {tic:6s} {px.dropna().index.min().date()} → "
              f"{px.dropna().index.max().date()}  ({px.dropna().size} obs)")

    prices = pd.concat(frames.values(), axis=1).dropna(how="all").ffill().dropna()
    rets = 100 * np.log(prices).diff().dropna()
    prices.to_csv(PROC / "prices_real.csv")
    rets.to_csv(PROC / "returns_real.csv")

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

    print("\nNOTE: for monthly inflation, the cedi reference rate and export "
          "receipts, download the Bank of Ghana time-series workbooks and "
          "save them under data/raw/ — see README §Data.")


if __name__ == "__main__":
    main()
