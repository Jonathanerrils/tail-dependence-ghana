import sys, pickle
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from tailrisk import copulas, inference

rets = pd.read_csv("data/processed/returns_real.csv", index_col=0, parse_dates=True)
cedi = rets["cedi"].dropna()

# ------------------------------------------------ Part 1: dynamic zero-probability
# Use the identical zero-process specification search used by Treatment E.
# AR(1) and AR(2) are compared on the same sample, since df_logit requires
# both lags before either model is fitted.

I_t = (cedi == 0).astype(int)
I_lag1 = I_t.shift(1)
I_lag2 = I_t.shift(2)

df_logit = pd.DataFrame({
    "I": I_t,
    "I1": I_lag1,
    "I2": I_lag2,
}).dropna()

logit1 = sm.Logit(
    df_logit["I"],
    sm.add_constant(df_logit[["I1"]]),
).fit(disp=0)

logit2 = sm.Logit(
    df_logit["I"],
    sm.add_constant(df_logit[["I1", "I2"]]),
).fit(disp=0)

best_logit, order = (
    (logit1, 1)
    if logit1.aic < logit2.aic
    else (logit2, 2)
)

print("Zero-indicator logistic model selection:")
print(f"  AR(1): AIC={logit1.aic:.3f}")
print(f"  AR(2): AIC={logit2.aic:.3f}")
print(f"  Selected AR({order})")

if order == 1:
    X = sm.add_constant(df_logit[["I1"]])
else:
    X = sm.add_constant(df_logit[["I1", "I2"]])

p_t = best_logit.predict(X)
p_t.name = "p_zero"

# ------------------------------------------------ Part 2: two-regime mixture
nz = cedi[cedi != 0]
X_nz = nz.to_numpy().reshape(-1, 1)

gm2 = GaussianMixture(
    n_components=2,
    covariance_type="full",
    n_init=20,
    random_state=0,
    max_iter=1000,
).fit(X_nz)

if not gm2.converged_:
    raise RuntimeError(
        "Treatment F two-regime GaussianMixture did not converge; "
        "cedi_hurdle_mixture_pit.csv will not be written."
    )

if not (
    np.all(np.isfinite(gm2.weights_))
    and np.all(np.isfinite(gm2.means_))
    and np.all(np.isfinite(gm2.covariances_))
):
    raise RuntimeError(
        "Treatment F two-regime GaussianMixture returned non-finite "
        "weights, means, or covariance estimates."
    )

# Single-Gaussian fit is used only for the descriptive BIC comparison,
# but verify it converged before reporting that comparison.
gm1 = GaussianMixture(n_components=1).fit(X_nz)
if not gm1.converged_:
    raise RuntimeError(
        "Treatment F single-Gaussian comparison model did not converge; "
        "BIC comparison is not valid."
    )

w = gm2.weights_
mu = gm2.means_.ravel()
sig = np.sqrt(gm2.covariances_.ravel())
quiet, volatile = (0, 1) if sig[0] < sig[1] else (1, 0)
print(f"Two-regime mixture fit to {len(nz)} nonzero cedi returns:")
print(f"  Quiet regime:    weight={w[quiet]:.3f}  mean={mu[quiet]:.4f}  std={sig[quiet]:.4f}")
print(f"  Volatile regime: weight={w[volatile]:.3f}  mean={mu[volatile]:.4f}  std={sig[volatile]:.4f}")
print(f"  BIC (k=2) vs single-Gaussian (k=1): "
      f"{gm2.bic(X_nz):.1f} vs {gm1.bic(X_nz):.1f}")


def mixture_cdf(x: np.ndarray) -> np.ndarray:
    return w[0] * stats.norm.cdf(x, mu[0], sig[0]) + w[1] * stats.norm.cdf(x, mu[1], sig[1])


# ------------------------------------------------ Part 3: mixture PIT construction
common_idx = p_t.index.intersection(cedi.index)
p_t2 = p_t.loc[common_idx]
r = cedi.loc[common_idx]

F0 = np.full(len(common_idx), mixture_cdf(np.array([0.0]))[0])
print(f"\nF0 (two-regime mixture CDF at the zero point): {F0[0]:.4f}")

rng = np.random.default_rng(7)
u_mix = np.empty(len(r))
is_zero = (r.to_numpy() == 0)
is_pos = (r.to_numpy() > 0)
u_mix[is_zero] = ((1 - p_t2.to_numpy()[is_zero]) * F0[is_zero]
                  + p_t2.to_numpy()[is_zero] * rng.uniform(0, 1, is_zero.sum()))
Fz = mixture_cdf(r.to_numpy())
u_mix[~is_zero & is_pos] = ((1 - p_t2.to_numpy()[~is_zero & is_pos]) * Fz[~is_zero & is_pos]
                            + p_t2.to_numpy()[~is_zero & is_pos])
mask_neg = ~is_zero & ~is_pos
u_mix[mask_neg] = (1 - p_t2.to_numpy()[mask_neg]) * Fz[mask_neg]

u_mix = np.clip(u_mix, 1e-9, 1 - 1e-9)
cedi_F = pd.Series(u_mix, index=common_idx, name="cedi")

print(f"\nu_mix histogram (10 bins):")
counts, edges = np.histogram(u_mix, bins=10, range=(0, 1))
for c, e in zip(counts, edges):
    print(f"  [{e:.1f}, {e+0.1:.1f}): {c}  (expected ~{len(u_mix)/10:.0f} if uniform)")

_pit_canonical = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
class _FitStub:
    def __init__(self, pit): self.pit = pit
fits = {c: _FitStub(_pit_canonical[c]) for c in _pit_canonical.columns}
ks_A = stats.kstest(fits["cedi"].pit.dropna(), "uniform")
ks_F = stats.kstest(cedi_F, "uniform")
print(f"\nKS test of PIT uniformity:")
print(f"  Treatment A (naive continuous, single-t):        p={ks_A.pvalue:.2e}")
print(f"  Treatment F (hurdle + two-regime mixture):       p={ks_F.pvalue:.4f}")

cedi_F.to_frame().to_csv("outputs/tables/cedi_hurdle_mixture_pit.csv")