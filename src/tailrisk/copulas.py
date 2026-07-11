"""Bivariate copula estimation and tail-dependence measurement.

Implements canonical-MLE on rank-based pseudo-observations for the
Gaussian, Student-t, Clayton (lower-tail) and Gumbel (upper-tail)
families, with analytic tail-dependence coefficients and a
nonparametric empirical estimator for model-free comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

EPS = 1e-9


def pseudo_obs(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Rank-based pseudo-observations u_i = rank_i / (n + 1)."""
    a = np.asarray(x, dtype=float)
    n = a.shape[0]
    ranks = np.apply_along_axis(stats.rankdata, 0, a)
    return ranks / (n + 1.0)


# ---------------------------------------------------------------- densities

def _gaussian_loglik(rho: float, z1: np.ndarray, z2: np.ndarray) -> float:
    r2 = rho * rho
    ll = -0.5 * np.log(1 - r2) - (r2 * (z1**2 + z2**2) - 2 * rho * z1 * z2) / (
        2 * (1 - r2)
    )
    return float(np.sum(ll))


def _t_loglik(rho: float, nu: float, u1: np.ndarray, u2: np.ndarray) -> float:
    x = stats.t.ppf(u1, df=nu)
    y = stats.t.ppf(u2, df=nu)
    r2 = rho * rho
    from scipy.special import gammaln

    q = (x**2 - 2 * rho * x * y + y**2) / (nu * (1 - r2))
    ll = (
        gammaln((nu + 2) / 2)
        + gammaln(nu / 2)
        - 2 * gammaln((nu + 1) / 2)
        - 0.5 * np.log(1 - r2)
        - ((nu + 2) / 2) * np.log1p(q)
        + ((nu + 1) / 2) * (np.log1p(x**2 / nu) + np.log1p(y**2 / nu))
    )
    return float(np.sum(ll))


def _clayton_loglik(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    s = u ** (-theta) + v ** (-theta) - 1.0
    ll = (
        np.log1p(theta)
        - (theta + 1) * (np.log(u) + np.log(v))
        - (2 + 1 / theta) * np.log(s)
    )
    return float(np.sum(ll))


def _gumbel_loglik(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    x = -np.log(u)
    y = -np.log(v)
    a = x**theta + y**theta
    w = a ** (1 / theta)
    # density: exp(-w) * (xy)^{theta-1} / (uv) * a^{2/theta - 2} * (1 + (theta-1)/w)
    ll = (
        -w
        + (theta - 1) * (np.log(x) + np.log(y))
        - (np.log(u) + np.log(v))
        + (2 / theta - 2) * np.log(a)
        + np.log1p((theta - 1) / w)
    )
    return float(np.sum(ll))


# ---------------------------------------------------------------- fitting

@dataclass
class CopulaFit:
    family: str
    params: dict = field(default_factory=dict)
    loglik: float = np.nan
    n: int = 0

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def lambda_lower(self) -> float:
        if self.family == "clayton":
            return 2.0 ** (-1.0 / self.params["theta"])
        if self.family == "t":
            rho, nu = self.params["rho"], self.params["nu"]
            arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            return float(2 * stats.t.cdf(arg, df=nu + 1))
        return 0.0  # Gaussian and Gumbel have zero lower-tail dependence

    @property
    def lambda_upper(self) -> float:
        if self.family == "gumbel":
            return 2.0 - 2.0 ** (1.0 / self.params["theta"])
        if self.family == "t":
            return self.lambda_lower  # t copula is tail-symmetric
        return 0.0


def fit_gaussian(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    z1, z2 = stats.norm.ppf(u), stats.norm.ppf(v)
    res = optimize.minimize_scalar(
        lambda r: -_gaussian_loglik(np.tanh(r), z1, z2), bounds=(-5, 5), method="bounded"
    )
    rho = float(np.tanh(res.x))
    return CopulaFit("gaussian", {"rho": rho}, _gaussian_loglik(rho, z1, z2), u.size)


def fit_t(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    def neg(p):
        rho = np.tanh(p[0])
        nu = 2.05 + np.exp(p[1])
        return -_t_loglik(rho, nu, u, v)

    tau = stats.kendalltau(u, v).statistic
    rho0 = np.sin(np.pi * tau / 2)
    res = optimize.minimize(neg, x0=[np.arctanh(np.clip(rho0, -0.95, 0.95)), np.log(6.0)],
                            method="Nelder-Mead")
    rho = float(np.tanh(res.x[0]))
    nu = float(2.05 + np.exp(res.x[1]))
    return CopulaFit("t", {"rho": rho, "nu": nu}, -res.fun, u.size)


def fit_clayton(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    res = optimize.minimize_scalar(
        lambda p: -_clayton_loglik(np.exp(p), u, v), bounds=(-6, 3.5), method="bounded"
    )
    theta = float(np.exp(res.x))
    return CopulaFit("clayton", {"theta": theta}, _clayton_loglik(theta, u, v), u.size)


def fit_gumbel(u: np.ndarray, v: np.ndarray) -> CopulaFit:
    res = optimize.minimize_scalar(
        lambda p: -_gumbel_loglik(1.0 + np.exp(p), u, v), bounds=(-8, 3), method="bounded"
    )
    theta = float(1.0 + np.exp(res.x))
    return CopulaFit("gumbel", {"theta": theta}, _gumbel_loglik(theta, u, v), u.size)


FAMILIES = {
    "gaussian": fit_gaussian,
    "t": fit_t,
    "clayton": fit_clayton,
    "gumbel": fit_gumbel,
}


def fit_pair(u: np.ndarray, v: np.ndarray) -> dict[str, CopulaFit]:
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    return {name: fn(u, v) for name, fn in FAMILIES.items()}


# ------------------------------------------------- nonparametric estimator

def empirical_lambda_lower(u: np.ndarray, v: np.ndarray, q: float = 0.05) -> float:
    """lambda_L(q) = P(U <= q, V <= q) / q  (converges to lambda_L as q->0)."""
    joint = np.mean((u <= q) & (v <= q))
    return float(joint / q)


def empirical_lambda_upper(u: np.ndarray, v: np.ndarray, q: float = 0.95) -> float:
    joint = np.mean((u > q) & (v > q))
    return float(joint / (1 - q))
