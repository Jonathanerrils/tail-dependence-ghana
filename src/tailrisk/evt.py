"""Extreme value theory layer: peaks-over-threshold with the GPD.

Applied to the loss tail (negated returns or negated standardized
residuals). Includes POT-implied VaR/ES, the Hill estimator, and a
threshold-sensitivity scan, which the paper uses as a robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class POTFit:
    threshold: float
    quantile: float          # threshold quantile in the loss distribution
    xi: float                # GPD shape (tail index > 0 => heavy tail)
    beta: float              # GPD scale
    n_exceed: int
    n_total: int

    def var(self, alpha: float) -> float:
        """POT VaR at confidence alpha (e.g. 0.99) for the loss variable."""
        zeta = self.n_exceed / self.n_total
        return self.threshold + (self.beta / self.xi) * (
            ((1 - alpha) / zeta) ** (-self.xi) - 1
        )

    def es(self, alpha: float) -> float:
        """POT expected shortfall at confidence alpha (requires xi < 1)."""
        v = self.var(alpha)
        return v / (1 - self.xi) + (self.beta - self.xi * self.threshold) / (1 - self.xi)


def fit_pot(losses: np.ndarray | pd.Series, quantile: float = 0.90) -> POTFit:
    """Fit a GPD to exceedances of the `quantile`-level threshold."""
    x = np.asarray(pd.Series(losses).dropna(), dtype=float)
    u = float(np.quantile(x, quantile))
    exceed = x[x > u] - u
    xi, _, beta = stats.genpareto.fit(exceed, floc=0.0)
    return POTFit(
        threshold=u,
        quantile=quantile,
        xi=float(xi),
        beta=float(beta),
        n_exceed=int(exceed.size),
        n_total=int(x.size),
    )


def threshold_sensitivity(
    losses: np.ndarray | pd.Series,
    quantiles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Shape estimate across thresholds — stability check for POT."""
    quantiles = quantiles if quantiles is not None else np.arange(0.85, 0.976, 0.005)
    rows = []
    for q in quantiles:
        f = fit_pot(losses, quantile=float(q))
        rows.append({"quantile": q, "threshold": f.threshold,
                     "xi": f.xi, "beta": f.beta, "n_exceed": f.n_exceed})
    return pd.DataFrame(rows)


def hill_estimator(losses: np.ndarray | pd.Series, k: int) -> float:
    """Hill tail-index estimator using the top-k order statistics.

    Returns gamma = 1/alpha (comparable to GPD xi for heavy tails).
    """
    x = np.sort(np.asarray(pd.Series(losses).dropna(), dtype=float))
    x = x[x > 0]
    top = x[-(k + 1):]
    return float(np.mean(np.log(top[1:] / top[0])))


def mean_excess(losses: np.ndarray | pd.Series, n_points: int = 60) -> pd.DataFrame:
    """Mean-excess function e(u) = E[X - u | X > u] on a grid of thresholds."""
    x = np.asarray(pd.Series(losses).dropna(), dtype=float)
    grid = np.quantile(x, np.linspace(0.70, 0.985, n_points))
    rows = []
    for u in grid:
        exc = x[x > u] - u
        if exc.size >= 10:
            rows.append({"threshold": u, "mean_excess": exc.mean(), "n": exc.size})
    return pd.DataFrame(rows)
