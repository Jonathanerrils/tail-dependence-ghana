"""Marginal models: AR(1)-GJR-GARCH(1,1) with Student-t innovations.

Each series is filtered through a location-scale model so that the copula
layer operates on (approximately) i.i.d. probability-integral-transformed
residuals, as required by Sklar-based two-step estimation (IFM).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats


@dataclass
class MarginalFit:
    name: str
    params: pd.Series
    nu: float                    # Student-t degrees of freedom
    std_resid: pd.Series         # standardized residuals
    pit: pd.Series               # PIT(u) = F_t(std_resid; nu)
    cond_vol: pd.Series
    aic: float
    bic: float


def fit_marginal(returns: pd.Series, name: str | None = None) -> MarginalFit:
    """Fit AR(1)-GJR-GARCH(1,1)-t to a percent-return series."""
    name = name or str(returns.name)
    r = returns.dropna()
    am = arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t")
    res = am.fit(disp="off")
    std_resid = (res.resid / res.conditional_volatility).dropna()
    nu = float(res.params["nu"])
    # PIT with the fitted standardized-t distribution (unit variance scaling)
    scale = np.sqrt(nu / (nu - 2.0))
    pit = pd.Series(
        stats.t.cdf(std_resid * scale, df=nu),
        index=std_resid.index,
        name=name,
    ).clip(1e-6, 1 - 1e-6)
    return MarginalFit(
        name=name,
        params=res.params,
        nu=nu,
        std_resid=std_resid.rename(name),
        pit=pit,
        cond_vol=res.conditional_volatility.rename(name),
        aic=res.aic,
        bic=res.bic,
    )


def fit_all(returns: pd.DataFrame) -> dict[str, MarginalFit]:
    return {c: fit_marginal(returns[c], c) for c in returns.columns}


def pit_frame(fits: dict[str, MarginalFit]) -> pd.DataFrame:
    """Aligned PIT matrix (drops dates lost to AR initialization)."""
    return pd.concat([f.pit for f in fits.values()], axis=1).dropna()


def std_resid_frame(fits: dict[str, MarginalFit]) -> pd.DataFrame:
    return pd.concat([f.std_resid for f in fits.values()], axis=1).dropna()
