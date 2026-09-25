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
    nu: float                    # Student-t df; NaN for non-Student-t innovation distributions
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
        eta = float(res.params["eta"])
        lam = float(res.params["lambda"])
        dist = res.model.distribution
        pit_vals = dist.cdf(std_resid.to_numpy(), parameters=[eta, lam])
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

    A specification passes if the Ljung-Box tests on standardized residuals
    and squared standardized residuals, together with the KS uniformity test
    on the PIT, all exceed `alpha`.

    If no specification passes, no candidate is declared preferred solely
    on the basis of AIC. The simplest successfully fitted ladder
    specification is retained as a deterministic reference model and
    labeled UNRESOLVED. Candidate AIC values remain available descriptively
    in the search log. Substantive inference for an unresolved marginal
    should be evaluated through the corresponding robustness analysis.
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
    # No candidate passed the adequacy gate. AIC-based selection among
    # inadequate candidates was tested for reproducibility and found NOT
    # to be stable across computational environments for at least one
    # series (the cedi) -- see diagnostics/diagnose_cedi_v3.py and
    # DECISIONS.md. Treating an unstable AIC comparison as though it
    # identifies a genuine "best" fallback would misrepresent what the
    # data actually supports. Instead: fall back deterministically to the
    # first (simplest) ladder specification, for procedural consistency
    # only, not as a claim that it is preferred. Every attempted
    # candidate's AIC is still reported descriptively in `log`.
    # Substantive conclusions involving this series should rely on the
    # alternative-treatment robustness analysis, not on this fallback fit.
    if not candidates:
        raise RuntimeError(f"No specification in SPEC_LADDER could be fit for {name}")

    baseline_name = SPEC_LADDER[0][0]
    fallback = next(
        ((f, sn, d) for f, sn, d in candidates if sn == baseline_name),
        candidates[0],
    )
    fit, actual_spec_name, diag = fallback

    label = (f"{actual_spec_name} (UNRESOLVED -- no candidate specification "
             f"passed the adequacy gate; AIC ranking among inadequate "
             f"candidates is not used to select a fallback; this "
             f"specification is retained only as a deterministic reference "
             f"for the downstream robustness analysis)")
    return fit, label, diag, log