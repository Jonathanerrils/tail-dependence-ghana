"""Grid-evaluated Rosenblatt Cramér-von Mises-type copula goodness-of-fit,
calibrated by parametric bootstrap.

Implementation notes (documented in DECISIONS.md):
* Statistic: under the fitted copula, the Rosenblatt-transformed pair
  (e1, e2) = (u, h(v|u)) should be i.i.d. Uniform(0,1)^2 if the family
  is correctly specified. The statistic measures the squared
  discrepancy, on a fixed 50x50 grid, between the empirical joint CDF
  of the transformed observations and the independence-copula CDF (uv)
  -- not a comparison against a simulated approximation of the fitted
  copula's own CDF, which an earlier version of this docstring
  incorrectly described. Not claimed here to be numerically identical
  to the Genest-Remillard-Beaudoin (2009) statistic without that
  equivalence having been separately verified; described conservatively
  as a Rosenblatt CvM-type statistic in this family's spirit.
* Estimator (data and bootstrap alike): the same continuous pseudo-MLE
  fitting functions used by the AIC-selection stage (src/tailrisk/
  copulas.py, FAMILIES dict), not a separate closed-form/tau-inversion
  estimator. This was previously a real methodological inconsistency --
  the manuscript describes fitting by pseudo-MLE and testing goodness-
  of-fit of those same fitted models, but this module's bootstrap
  actually used Kendall-tau inversion internally, a different (though
  individually valid) estimator. Unified so the GOF stage genuinely
  tests the models the AIC stage selected among.
* p-value: (exceed + 1) / (n_boot + 1), the conventional finite-bootstrap
  correction, consistent with the convention used in inference.py's
  stress-test p-values. An earlier version used a "+0.5" mid-p-style
  correction here with no stated justification for why this test
  specifically should differ from that convention.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .copulas import FAMILIES


# ------------------------------------------------------------- estimators
def estimate(family: str, u: np.ndarray, v: np.ndarray) -> dict:
    """Fit `family` to (u, v) using the exact same pseudo-MLE function
    the AIC-selection stage uses, so the GOF bootstrap tests the same
    class of fitted model throughout, not a separate closed-form
    approximation to it."""
    return FAMILIES[family](u, v).params


# --------------------------------------------------------------- samplers
def sample(family: str, params: dict, n: int,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if family == "gaussian":
        rho = params["rho"]
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(
            [[1, rho], [rho, 1]]).T
        return stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
    if family == "t":
        rho, nu = params["rho"], params["nu"]
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(
            [[1, rho], [rho, 1]]).T
        w = rng.chisquare(nu, n) / nu
        t = z / np.sqrt(w)[:, None]
        return stats.t.cdf(t[:, 0], nu), stats.t.cdf(t[:, 1], nu)
    if family == "clayton":
        th = params["theta"]
        g = rng.gamma(1 / th, 1.0, n)
        e = rng.exponential(1.0, (n, 2))
        u = (1 + e / g[:, None]) ** (-1 / th)
        return u[:, 0], u[:, 1]
    if family == "gumbel":
        th = params["theta"]
        alpha = 1 / th
        # positive stable via Chambers-Mallows-Stuck
        t_ = rng.uniform(0, np.pi, n)
        w_ = rng.exponential(1.0, n)
        s = (np.sin(alpha * t_) / np.sin(t_) ** (1 / alpha)
             * (np.sin((1 - alpha) * t_) / w_) ** ((1 - alpha) / alpha))
        e = rng.exponential(1.0, (n, 2))
        u = np.exp(-((e / s[:, None]) ** (1 / th)))
        return u[:, 0], u[:, 1]
    if family == "frank":
        th = params["theta"]
        u1 = rng.uniform(0, 1, n)
        w = rng.uniform(0, 1, n)
        if abs(th) < 1e-6:
            return u1, w
        eu = np.exp(th * u1)
        e_th = np.exp(th)
        num = w * (eu - 1) + 1
        den = e_th + w * (eu - e_th)
        u2 = (th + np.log(num / den)) / th
        return u1, np.clip(u2, 1e-9, 1 - 1e-9)
    if family == "joe":
        # Conditional (h-function inversion) sampler, vectorized across all
        # n observations at once via array-valued bisection -- exact for
        # any Archimedean copula with a tractable h-function, and fast
        # enough for the bootstrap this feeds.
        th = params["theta"]
        u1 = rng.uniform(0, 1, n)
        w = rng.uniform(0, 1, n)
        lo = np.full(n, 1e-9)
        hi = np.full(n, 1 - 1e-9)
        flo = h_func("joe", {"theta": th}, u1, lo) - w
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fmid = h_func("joe", {"theta": th}, u1, mid) - w
            go_right = (fmid * flo) > 0
            lo = np.where(go_right, mid, lo)
            flo = np.where(go_right, fmid, flo)
            hi = np.where(go_right, hi, mid)
        return u1, 0.5 * (lo + hi)
    if family == "survival_clayton":
        th = params["theta"]
        g = rng.gamma(1 / th, 1.0, n)
        e = rng.exponential(1.0, (n, 2))
        u = (1 + e / g[:, None]) ** (-1 / th)
        return 1 - u[:, 0], 1 - u[:, 1]
    if family == "survival_gumbel":
        th = params["theta"]
        alpha = 1 / th
        t_ = rng.uniform(0, np.pi, n)
        w_ = rng.exponential(1.0, n)
        s = (np.sin(alpha * t_) / np.sin(t_) ** (1 / alpha)
             * (np.sin((1 - alpha) * t_) / w_) ** ((1 - alpha) / alpha))
        e = rng.exponential(1.0, (n, 2))
        u = np.exp(-((e / s[:, None]) ** (1 / th)))
        return 1 - u[:, 0], 1 - u[:, 1]
    raise ValueError(family)


# ------------------------------------------------------------- statistic
def _pseudo(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x) / (len(x) + 1)


def h_func(family: str, params: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Conditional CDF h(v|u) = dC(u,v)/du — exact for all supported families."""
    u = np.clip(u, 1e-9, 1 - 1e-9)
    v = np.clip(v, 1e-9, 1 - 1e-9)
    if family == "gaussian":
        rho = params["rho"]
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        return stats.norm.cdf((y - rho * x) / np.sqrt(1 - rho**2))
    if family == "t":
        rho, nu = params["rho"], params["nu"]
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        z = (y - rho * x) / np.sqrt((1 - rho**2) * (nu + x**2) / (nu + 1))
        return stats.t.cdf(z, nu + 1)
    if family == "clayton":
        th = params["theta"]
        return u ** (-th - 1) * (u ** (-th) + v ** (-th) - 1) ** (-1 / th - 1)
    if family == "gumbel":
        th = params["theta"]
        x, y = -np.log(u), -np.log(v)
        a = x**th + y**th
        return np.exp(-a ** (1 / th)) * a ** (1 / th - 1) * x ** (th - 1) / u
    if family == "frank":
        th = params["theta"]
        if abs(th) < 1e-6:
            return v  # independence limit
        eu = np.exp(-th * u)
        num = np.expm1(-th * v) * eu
        den = np.expm1(-th) + np.expm1(-th * u) * np.expm1(-th * v)
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return np.clip(num / den, 1e-12, 1 - 1e-12)
    if family == "joe":
        th = params["theta"]
        ub, vb = 1 - u, 1 - v
        t1, t2 = ub**th, vb**th
        a = t1 + t2 - t1 * t2
        return np.clip(a ** (1 / th - 1) * ub ** (th - 1) * (1 - t2), 1e-12, 1 - 1e-12)
    if family == "survival_clayton":
        return 1 - h_func("clayton", params, 1 - u, 1 - v)
    if family == "survival_gumbel":
        return 1 - h_func("gumbel", params, 1 - u, 1 - v)
    raise ValueError(family)


def cvm_rosenblatt(u: np.ndarray, v: np.ndarray, family: str, params: dict,
                   grid_size: int = 50) -> float:
    """Grid-evaluated Rosenblatt CvM-type statistic.

    Under H0, e1=u and e2=h(v|u) should follow the independence
    copula. The statistic measures squared empirical-CDF discrepancy
    from uv over a fixed grid.
    """
    e1, e2 = u, h_func(family, params, u, v)
    g = np.arange(1, grid_size + 1) / (grid_size + 1)
    a = (e1[:, None] <= g[None, :]).astype(np.float32)
    b = (e2[:, None] <= g[None, :]).astype(np.float32)
    d_emp = (a.T @ b) / len(u)
    pi = g[:, None] * g[None, :]
    return float(np.sum((d_emp - pi) ** 2))


def gof_test(u: np.ndarray, v: np.ndarray, family: str,
             n_boot: int = 500, grid_size: int = 50,
             seed: int = 0) -> dict:
    """Parametric-bootstrap p-value for H0: copula belongs to `family`."""
    rng = np.random.default_rng(seed)
    n = len(u)
    theta0 = estimate(family, u, v)
    s0 = cvm_rosenblatt(u, v, family, theta0, grid_size)
    exceed = 0
    for _ in range(n_boot):
        su, sv = sample(family, theta0, n, rng)
        su, sv = _pseudo(su), _pseudo(sv)
        th_b = estimate(family, su, sv)
        exceed += (cvm_rosenblatt(su, sv, family, th_b, grid_size) >= s0)
    return {"family": family, "params": theta0, "Sn": s0,
            "p_value": (exceed + 1) / (n_boot + 1), "n_boot": n_boot}