"""Generate a synthetic demo dataset (2015-01-02 to 2026-06-30).

The generator embeds the stylized facts the paper tests for, so the
pipeline can be validated end-to-end before real data is plugged in:

* GARCH(1,1) volatility clustering with Student-t shocks per asset;
* a regime-switching t-copula: moderate correlation and high d.o.f. in
  calm periods, high correlation and low d.o.f. (strong tail dependence)
  in stress windows (COVID Mar–Jun 2020; the 2024 cocoa shock);
* a cocoa-specific 2024 episode: large positive drift + a volatility
  explosion, mimicking the historic 2024 rally;
* a Ghana cedi (GHS/USD) series whose crash risk loads on commodity
  lower-tail events, so commodity co-crashes transmit to FX losses.

THIS IS SYNTHETIC DATA. Use scripts/01_fetch_real_data.py locally to
replace it with real market data before drawing any empirical claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

ASSETS = ["cocoa", "gold", "brent", "wti", "ghs_usd"]

# calm-regime correlation (roughly plausible cross-commodity structure)
R_CALM = np.array([
    #  cocoa  gold  brent  wti   ghs
    [1.00, 0.08, 0.12, 0.11, 0.10],
    [0.08, 1.00, 0.18, 0.17, 0.15],
    [0.12, 0.18, 1.00, 0.92, 0.20],
    [0.11, 0.17, 0.92, 1.00, 0.19],
    [0.10, 0.15, 0.20, 0.19, 1.00],
])

# stress regime: broad co-movement rises, cedi loads on commodities
R_STRESS = np.array([
    [1.00, 0.30, 0.45, 0.44, 0.40],
    [0.30, 1.00, 0.40, 0.39, 0.35],
    [0.45, 0.40, 1.00, 0.95, 0.50],
    [0.44, 0.39, 0.95, 1.00, 0.49],
    [0.40, 0.35, 0.50, 0.49, 1.00],
])

# per-asset GARCH(1,1) parameters: (omega, alpha, beta, nu, ann. drift %)
GARCH = {
    "cocoa":   (0.06, 0.09, 0.88, 5.0,  4.0),
    "gold":    (0.01, 0.06, 0.92, 7.0,  6.0),
    "brent":   (0.05, 0.10, 0.87, 5.5,  2.0),
    "wti":     (0.06, 0.11, 0.86, 5.0,  2.0),
    "ghs_usd": (0.02, 0.12, 0.85, 4.5, 14.0),  # cedi depreciation drift
}


def stress_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    covid = (dates >= "2020-03-01") & (dates <= "2020-06-30")
    cocoa24 = (dates >= "2024-01-01") & (dates <= "2024-12-31")
    return np.asarray(covid | cocoa24)


def main() -> None:
    dates = pd.bdate_range("2015-01-02", "2026-06-30")
    T, k = len(dates), len(ASSETS)
    stress = stress_mask(dates)

    # --- regime-dependent t-copula innovations -> uniform PITs
    u = np.empty((T, k))
    for regime, R, nu in [(False, R_CALM, 12.0), (True, R_STRESS, 4.0)]:
        idx = np.where(stress == regime)[0]
        L = np.linalg.cholesky(R)
        z = RNG.standard_normal((idx.size, k)) @ L.T
        w = RNG.chisquare(nu, size=(idx.size, 1)) / nu
        t_draws = z / np.sqrt(w)
        u[idx] = stats.t.cdf(t_draws, df=nu)

    # --- GARCH filtering per asset
    rets = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
    for j, a in enumerate(ASSETS):
        omega, alpha, beta, nu, drift = GARCH[a]
        scale = np.sqrt(nu / (nu - 2.0))
        eps = stats.t.ppf(u[:, j], df=nu) / scale  # unit-variance t shocks
        h = np.empty(T)
        h[0] = omega / (1 - alpha - beta)
        r = np.empty(T)
        mu = drift / 252.0 / 100.0 * 100.0  # daily % drift
        for t in range(T):
            if t > 0:
                h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
            r[t] = mu / 100 + np.sqrt(h[t]) * eps[t]
        rets[a] = r

    # --- 2024 cocoa episode: drift + vol amplification
    cocoa24 = (dates >= "2024-01-01") & (dates <= "2024-12-31")
    rally = (dates >= "2024-01-01") & (dates <= "2024-04-30")
    rets.loc[cocoa24, "cocoa"] *= 2.2
    rets.loc[rally, "cocoa"] += 0.55  # strong upward drift, % per day

    # cedi: commodity co-crash days add FX losses (transmission channel)
    co_crash = (rets["cocoa"] < rets["cocoa"].quantile(0.05)) & (
        rets["brent"] < rets["brent"].quantile(0.05)
    )
    rets.loc[co_crash, "ghs_usd"] += RNG.normal(1.2, 0.4, co_crash.sum())

    # Orient the cedi like the real-data pipeline: flip GHS-per-USD return
    # sign and rename to 'cedi', so lower tail = depreciation for all series.
    rets["ghs_usd"] = -rets["ghs_usd"]
    rets = rets.rename(columns={"ghs_usd": "cedi"})

    prices = 100 * np.exp(rets.div(100).cumsum())
    prices.to_csv(OUT / "prices_synthetic.csv")
    rets.to_csv(OUT / "returns_synthetic.csv")

    # --- monthly Ghana macro (synthetic): inflation & export revenue proxy
    m_idx = pd.period_range("2015-01", "2026-06", freq="M").to_timestamp("M")
    infl = 12 + 8 * np.sin(np.arange(len(m_idx)) / 14) + RNG.normal(0, 1.2, len(m_idx))
    infl += 10 * ((m_idx >= "2022-06") & (m_idx <= "2023-06"))  # 2022 crisis bump
    cocoa_m = prices["cocoa"].resample("ME").last().reindex(m_idx).ffill()
    gold_m = prices["gold"].resample("ME").last().reindex(m_idx).ffill()
    exports = 0.5 * cocoa_m / cocoa_m.iloc[0] + 0.5 * gold_m / gold_m.iloc[0]
    exports = 1500 * exports * np.exp(RNG.normal(0, 0.03, len(m_idx)))
    macro = pd.DataFrame(
        {"cpi_inflation_yoy": infl, "export_revenue_usd_m": exports.to_numpy()},
        index=m_idx,
    )
    macro.to_csv(OUT / "ghana_macro_synthetic.csv")

    print(f"Wrote {T} daily obs for {k} series and {len(m_idx)} monthly macro obs.")
    print("Stress days:", int(stress.sum()))


if __name__ == "__main__":
    main()
