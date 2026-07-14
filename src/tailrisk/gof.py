"""Cramér-von Mises copula goodness-of-fit with parametric bootstrap
(Genest, Rémillard & Beaudoin 2009 procedure, grid-evaluated variant).

Implementation notes (documented in DECISIONS.md):
* Statistic: Sn = sum over a fixed 50x50 grid of (C_n(u) - C_theta(u))^2,
  where C_n is the empirical copula of the pseudo-observations and
  C_theta the fitted copula's CDF approximated by the empirical CDF of
  m=5000 simulated points. Using the SAME statistic and the SAME
  estimator for the data and every bootstrap replicate makes the
  p-value valid; the grid/simulation approximation trades a little
  power for tractability and is standard practice.
* Estimator (data and bootstrap alike): Kendall-tau inversion for the
  dependence parameter (rho = sin(pi*tau/2) for elliptical families,
  theta = 2tau/(1-tau) Clayton, theta = 1/(1-tau) Gumbel), with the
  t copula's nu profiled over a fixed grid by pseudo-likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .copulas import _t_loglik

NU_GRID = np.array([3, 4, 5, 6, 8, 12, 20, 40], dtype=float)


# ------------------------------------------------------------- estimators
def estimate(family: str, u: np.ndarray, v: np.ndarray) -> dict:
    tau = stats.kendalltau(u, v).statistic
    tau = float(np.clip(tau, -0.95, 0.95))
    if family == "gaussian":
        return {"rho": np.sin(np.pi * tau / 2)}
    if family == "t":
        rho = np.sin(np.pi * tau / 2)
        lls = [_t_loglik(rho, nu, u, v) for nu in NU_GRID]
        return {"rho": rho, "nu": float(NU_GRID[int(np.argmax(lls))])}
    if family == "clayton":
        return {"theta": max(2 * tau / (1 - tau), 1e-3)}
    if family == "gumbel":
        return {"theta": max(1 / (1 - tau), 1.0 + 1e-6)}
    raise ValueError(family)


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
    raise ValueError(family)


# ------------------------------------------------------------- statistic
def _pseudo(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x) / (len(x) + 1)


def h_func(family: str, params: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Conditional CDF h(v|u) = dC(u,v)/du — exact for all four families."""
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
    raise ValueError(family)


def cvm_rosenblatt(u: np.ndarray, v: np.ndarray, family: str, params: dict,
                   grid_size: int = 50) -> float:
    """Sn(B) of Genest-Rémillard-Beaudoin (2009): after the Rosenblatt
    transform e1=u, e2=h(v|u), the pair is i.i.d. Uniform(0,1)^2 under H0;
    Sn(B) is the CvM distance of its empirical CDF from the independence
    copula, evaluated exactly on a grid."""
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
            "p_value": (exceed + 0.5) / (n_boot + 1), "n_boot": n_boot}
