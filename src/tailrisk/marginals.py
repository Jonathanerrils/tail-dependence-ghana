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
    return _fit_spec(returns, name or str(returns.name),
                     dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1,
                          dist="t"))


def _fit_spec(returns: pd.Series, name: str, spec: dict,
              spec_label: str = "AR(1)-GJR-GARCH(1,1)-t") -> MarginalFit:
    r = returns.dropna()
    res = arch_model(r, **spec).fit(disp="off")
    std_resid = (res.resid / res.conditional_volatility).dropna()
    dist = res.model.distribution
    k = dist.num_params
    dparams = res.params.values[-k:] if k else []
    pit = pd.Series(np.asarray(dist.cdf(std_resid.to_numpy(), dparams)),
                    index=std_resid.index, name=name).clip(1e-6, 1 - 1e-6)
    nu = float(res.params.get("nu", np.nan))
    fit = MarginalFit(name=name, params=res.params, nu=nu,
                      std_resid=std_resid.rename(name), pit=pit,
                      cond_vol=res.conditional_volatility.rename(name),
                      aic=res.aic, bic=res.bic)
    fit.spec = spec_label  # attached attribute: which rung of the ladder
    return fit


# Specification ladder for the adequacy gate (order fixed ex ante):
SPEC_LADDER = [
    ("AR(1)-GJR-GARCH(1,1)-t",
     dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR(2)-GJR-GARCH(1,1)-t",
     dict(mean="AR", lags=2, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR(1)-EGARCH(1,1)-t",
     dict(mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t")),
    ("AR(1)-GJR-GARCH(1,1)-skewt",
     dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt")),
]


def fit_with_gate(returns: pd.Series, name: str | None = None,
                  alpha: float = 0.05) -> tuple[MarginalFit, list[dict]]:
    """Walk SPEC_LADDER; return the first adequate fit (Ljung-Box on
    residuals and squares, KS uniformity of PIT, all p > alpha).
    If none passes, return the best-AIC fit flagged inadequate.
    Also returns the full search log for DECISIONS.md."""
    from .inference import marginal_diagnostics

    name = name or str(returns.name)
    log, fits = [], []
    for spec_label, spec in SPEC_LADDER:
        try:
            f = _fit_spec(returns, name, spec, spec_label)
            d = marginal_diagnostics(f.std_resid, f.pit)
        except Exception as e:  # noqa: BLE001
            log.append({"series": name, "spec": spec_label, "error": str(e)})
            continue
        log.append({"series": name, "spec": spec_label, "aic": f.aic, **d})
        fits.append((f, d))
        if d["adequate_5pct"]:
            f.adequate = True
            return f, log
    # none adequate: best AIC, flagged
    f = min((f for f, _ in fits), key=lambda x: x.aic)
    f.adequate = False
    return f, log


def fit_all(returns: pd.DataFrame, gate: bool = False):
    """Fit every column. gate=True walks the specification ladder and
    returns (fits, search_log); gate=False keeps the legacy behaviour."""
    if not gate:
        return {c: fit_marginal(returns[c], c) for c in returns.columns}
    fits, logs = {}, []
    for c in returns.columns:
        f, spec, diag, log = fit_with_gate(returns[c], c)
        f.spec = spec
        f.adequate = diag["adequate_5pct"]
        fits[c] = f
        logs.extend(log)
        print(f"  marginal {c}: {spec} (LBr={diag['lb_resid_p']:.3f}, "
              f"LBsq={diag['lb_sq_p']:.3f}, KS={diag['ks_pit_p']:.3f})")
    return fits, logs


def pit_frame(fits: dict[str, MarginalFit]) -> pd.DataFrame:
    """Aligned PIT matrix (drops dates lost to AR initialization)."""
    return pd.concat([f.pit for f in fits.values()], axis=1).dropna()


def std_resid_frame(fits: dict[str, MarginalFit]) -> pd.DataFrame:
    return pd.concat([f.std_resid for f in fits.values()], axis=1).dropna()


# ---------------------------------------------------------------------------
# Adequacy-gated fitting: try specs in order until diagnostics pass (5%).
# ---------------------------------------------------------------------------
SPEC_LADDER = [
    ("AR1-GJR-GARCH-t",     dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR2-GJR-GARCH-t",     dict(mean="AR", lags=2, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR1-EGARCH-t",        dict(mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t")),
    ("AR1-GJR-GARCH-skewt", dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt")),
]


def _fit_spec(returns: pd.Series, name: str, spec: dict) -> MarginalFit:
    r = returns.dropna()
    res = arch_model(r, **spec).fit(disp="off")
    std_resid = (res.resid / res.conditional_volatility).dropna()
    nu = float(res.params["nu"]) if "nu" in res.params else np.nan
    if spec["dist"] == "skewt":
        lam = float(res.params["lambda"])
        dist = res.model.distribution
        pit_vals = dist.cdf(std_resid.to_numpy(), parameters=[nu, lam])
    else:
        scale = np.sqrt(nu / (nu - 2.0))
        pit_vals = stats.t.cdf(std_resid * scale, df=nu)
    pit = pd.Series(pit_vals, index=std_resid.index, name=name).clip(1e-6, 1 - 1e-6)
    return MarginalFit(name=name, params=res.params, nu=nu,
                       std_resid=std_resid.rename(name), pit=pit,
                       cond_vol=res.conditional_volatility.rename(name),
                       aic=res.aic, bic=res.bic)


def fit_with_gate(returns: pd.Series, name: str | None = None,
                  alpha: float = 0.05):
    """Walk SPEC_LADDER; return (fit, spec_name, diagnostics, search_log).

    A spec passes if Ljung-Box on standardized residuals and squares and
    the KS uniformity test on the PIT all exceed `alpha`. If none passes,
    the best-AIC candidate is returned with adequate=False.
    """
    from .inference import marginal_diagnostics

    name = name or str(returns.name)
    log, candidates = [], []
    for spec_name, spec in SPEC_LADDER:
        try:
            fit = _fit_spec(returns, name, spec)
        except Exception as e:  # noqa: BLE001
            log.append({"series": name, "spec": spec_name, "error": str(e)})
            continue
        diag = marginal_diagnostics(fit.std_resid, fit.pit)
        log.append({"series": name, "spec": spec_name, "aic": fit.aic, **diag})
        candidates.append((fit, spec_name, diag))
        if diag["adequate_5pct"]:
            return fit, spec_name, diag, log
    best = min(candidates, key=lambda c: c[0].aic)
    return best[0], best[1] + " (INADEQUATE — best AIC)", best[2], log
