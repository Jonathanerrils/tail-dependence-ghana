"""Inference and diagnostics: what turns estimates into evidence.

1. Moving-block bootstrap confidence intervals for empirical tail
   dependence, per subsample (calm/stress) and for the calm-stress
   difference. PIT residuals are close to i.i.d. after GARCH filtering,
   but block resampling (Kunsch 1989) guards against residual serial
   dependence at no real cost.
2. Marginal adequacy diagnostics: Ljung-Box on standardized residuals
   and their squares (remaining ARMA/ARCH effects), and a
   Kolmogorov-Smirnov test of PIT uniformity. If these fail, the copula
   layer's inputs are suspect and the marginal spec needs revisiting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .copulas import empirical_lambda_lower


def _block_resample(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block bootstrap replicate of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def bootstrap_lambda_ci(
    u: np.ndarray, v: np.ndarray, q: float = 0.05,
    n_boot: int = 500, block: int = 20, level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Point estimate and percentile CI for lambda_L(q); returns the
    bootstrap draws too so differences can be built from them."""
    rng = rng or np.random.default_rng(0)
    n = len(u)
    block = max(2, min(block, n // 4))
    point = empirical_lambda_lower(u, v, q)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_resample(n, block, rng)
        ub = stats.rankdata(u[idx]) / (n + 1)   # re-rank within replicate
        vb = stats.rankdata(v[idx]) / (n + 1)
        draws[b] = empirical_lambda_lower(ub, vb, q)
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return point, float(lo), float(hi), draws


def calm_stress_difference(
    u_calm: np.ndarray, v_calm: np.ndarray,
    u_stress: np.ndarray, v_stress: np.ndarray,
    q: float = 0.10, n_boot: int = 500, level: float = 0.95,
    seed: int = 0, bonf_m: int | None = None, upper: bool = False,
) -> dict:
    """Bootstrap CI for lambda(stress) - lambda(calm). Subsamples are
    resampled independently (disjoint time windows). upper=True measures
    the upper tail via the survival transform (u,v) -> (1-u, 1-v).
    bonf_m adds a Bonferroni-adjusted CI for m simultaneous pairs."""
    if upper:
        u_calm, v_calm = 1 - u_calm, 1 - v_calm
        u_stress, v_stress = 1 - u_stress, 1 - v_stress
    rng = np.random.default_rng(seed)
    p_c, lo_c, hi_c, d_c = bootstrap_lambda_ci(u_calm, v_calm, q, n_boot, rng=rng)
    p_s, lo_s, hi_s, d_s = bootstrap_lambda_ci(u_stress, v_stress, q, n_boot, rng=rng)
    diff = d_s - d_c
    a = (1 - level) / 2
    dlo, dhi = np.quantile(diff, [a, 1 - a])
    out = {
        "lambda_calm": p_c, "calm_lo": lo_c, "calm_hi": hi_c,
        "lambda_stress": p_s, "stress_lo": lo_s, "stress_hi": hi_s,
        "difference": p_s - p_c, "diff_lo": float(dlo), "diff_hi": float(dhi),
        "significant_5pct": bool(dlo > 0 or dhi < 0),
    }
    if bonf_m:
        ab = (0.05 / bonf_m) / 2
        blo, bhi = np.quantile(diff, [ab, 1 - ab])
        out["diff_lo_bonf"] = float(blo)
        out["diff_hi_bonf"] = float(bhi)
        out["significant_bonferroni"] = bool(blo > 0 or bhi < 0)
    return out


def marginal_diagnostics(std_resid: pd.Series, pit: pd.Series,
                         lags: int = 10) -> dict:
    """Adequacy tests for one fitted marginal. All p-values: small = bad.

    lb_resid_p   Ljung-Box on standardized residuals (remaining ARMA)
    lb_sq_p      Ljung-Box on squared residuals (remaining ARCH)
    ks_pit_p     KS test of PIT against Uniform(0,1)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    z = std_resid.dropna().to_numpy()
    lb1 = acorr_ljungbox(z, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
    lb2 = acorr_ljungbox(z**2, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
    ks = stats.kstest(pit.dropna().to_numpy(), "uniform").pvalue
    return {"lb_resid_p": float(lb1), "lb_sq_p": float(lb2),
            "ks_pit_p": float(ks),
            "adequate_5pct": bool(lb1 > .05 and lb2 > .05 and ks > .05)}
