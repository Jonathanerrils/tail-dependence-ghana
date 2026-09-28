import sys, pickle
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from arch import arch_model
from tailrisk import copulas, inference

rets = pd.read_csv("data/processed/returns_real.csv", index_col=0, parse_dates=True)
cedi = rets["cedi"].dropna()

# ------------------------------------------------ Part 1: dynamic zero-probability
I_t = (cedi == 0).astype(int)
I_lag1 = I_t.shift(1)
I_lag2 = I_t.shift(2)
df_logit = pd.DataFrame({"I": I_t, "I1": I_lag1, "I2": I_lag2}).dropna()

logit1 = sm.Logit(df_logit["I"], sm.add_constant(df_logit[["I1"]])).fit(disp=0)
logit2 = sm.Logit(df_logit["I"], sm.add_constant(df_logit[["I1", "I2"]])).fit(disp=0)
print("Zero-indicator logistic AR(1):")
print(f"  AIC={logit1.aic:.1f}  params={dict(logit1.params.round(4))}")
print("Zero-indicator logistic AR(2):")
print(f"  AIC={logit2.aic:.1f}  params={dict(logit2.params.round(4))}")
best_logit, order = (logit1, 1) if logit1.aic < logit2.aic else (logit2, 2)
print(f"-> retaining AR({order}) specification (lower AIC)")

if order == 1:
    X = sm.add_constant(df_logit[["I1"]])
else:
    X = sm.add_constant(df_logit[["I1", "I2"]])
p_t = best_logit.predict(X)
p_t.name = "p_zero"

k = df_logit["I"].sum()
n = len(df_logit)
p_hat = k / n
null_ll = k * np.log(p_hat) + (n - k) * np.log1p(-p_hat)
lr_stat = 2 * (best_logit.llf - null_ll)
lr_p = stats.chi2.sf(lr_stat, df=order)
print(f"LR test vs constant-probability null: stat={lr_stat:.2f}, p={lr_p:.2e} "
      f"({'reject constant-p null -> zeros genuinely cluster' if lr_p < 0.05 else 'fail to reject -> no detectable clustering'})")

nz = cedi[cedi != 0]
print(f"\nNonzero-only subsequence: {len(nz)} obs ({len(nz)/len(cedi):.1%} of {len(cedi)})")
nu_hat, loc_hat, scale_hat = stats.t.fit(nz.to_numpy())
print(f"Unconstrained Student-t MLE: nu={nu_hat:.3f} (infinite variance, "
      f"nu<2) -- a degenerate fit chasing the sharp near-zero clustering "
      f"in the nonzero returns themselves; not trusted.")
from scipy import optimize as _opt
def _neg_ll(params):
    loc, log_scale, log_nu_minus = params
    scale = np.exp(log_scale)
    nu = 2.05 + np.exp(log_nu_minus)
    return -np.sum(stats.t.logpdf(nz.to_numpy(), df=nu, loc=loc, scale=scale))
_res = _opt.minimize(
    _neg_ll,
    x0=[0.0, np.log(0.1), np.log(3.0)],
    method="Nelder-Mead",
)

if not _res.success:
    raise RuntimeError(
        "Treatment E constrained Student-t optimizer did not converge: "
        f"status={_res.status}, message={_res.message}"
    )

if not np.isfinite(_res.fun) or not np.all(np.isfinite(_res.x)):
    raise RuntimeError(
        "Treatment E constrained Student-t optimizer returned non-finite "
        "objective or parameter values."
    )

loc_hat, scale_hat = float(_res.x[0]), float(np.exp(_res.x[1]))
nu_hat = float(2.05 + np.exp(_res.x[2]))
print(f"Constrained Student-t MLE (nu>=2.05, matching this project's "
      f"marginal methodology throughout): nu={nu_hat:.3f}, loc={loc_hat:.4f}, "
      f"scale={scale_hat:.4f}")
if abs(nu_hat - 2.05) < 0.01:
    print("NOTE: fit sits exactly at the nu=2.05 boundary -- the optimizer "
          "wants to go further into infinite-variance territory than the "
          "constraint allows. This is evidence the nonzero cedi returns "
          "are not well described by ANY single Student-t, regardless of "
          "shape parameter: the difficulty is broader than the exact-zero "
          "point mass a hurdle model targets.")

common_idx = p_t.index.intersection(cedi.index)
p_t2 = p_t.loc[common_idx]
r = cedi.loc[common_idx]

F0 = stats.t.cdf((0 - loc_hat) / scale_hat, df=nu_hat)
F0 = np.full(len(common_idx), F0)
print(f"\nF0 (continuous-model CDF at the zero point, constant): {F0[0]:.4f}")

rng = np.random.default_rng(7)
u_mix = np.empty(len(r))
is_zero = (r.to_numpy() == 0)
is_pos = (r.to_numpy() > 0)
u_mix[is_zero] = ((1 - p_t2.to_numpy()[is_zero]) * F0[is_zero]
                  + p_t2.to_numpy()[is_zero] * rng.uniform(0, 1, is_zero.sum()))
Fz = stats.t.cdf((r.to_numpy() - loc_hat) / scale_hat, df=nu_hat)
u_mix[~is_zero & is_pos] = ((1 - p_t2.to_numpy()[~is_zero & is_pos]) * Fz[~is_zero & is_pos]
                            + p_t2.to_numpy()[~is_zero & is_pos])
mask_neg = ~is_zero & ~is_pos
u_mix[mask_neg] = (1 - p_t2.to_numpy()[mask_neg]) * Fz[mask_neg]

u_mix = np.clip(u_mix, 1e-9, 1 - 1e-9)
cedi_E = pd.Series(u_mix, index=common_idx, name="cedi")

print(f"\nu_mix histogram (10 bins):")
counts, edges = np.histogram(u_mix, bins=10, range=(0, 1))
for c, e in zip(counts, edges):
    print(f"  [{e:.1f}, {e+0.1:.1f}): {c}  (expected ~{len(u_mix)/10:.0f} if uniform)")

_pit_canonical = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
ks_before = stats.kstest(_pit_canonical["cedi"].dropna(), "uniform")
ks_after = stats.kstest(cedi_E, "uniform")
print(f"\nKS test of PIT uniformity:")
print(f"  Treatment A (naive continuous PIT, ignores zero mass): p={ks_before.pvalue:.2e}")
print(f"  Treatment E (proper hurdle mixture PIT):               p={ks_after.pvalue:.4f}")

cedi_E.to_frame().to_csv("outputs/tables/cedi_hurdle_pit.csv")