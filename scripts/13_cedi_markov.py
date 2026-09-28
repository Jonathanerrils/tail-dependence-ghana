import sys, pickle
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM
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

# ------------------------------------------------ Part 2: 2-state Markov-switching
# continuous component, fit on the nonzero subsequence (same compression
# simplification as the static-mixture treatment: transition probabilities
# apply between consecutive NONZERO observations, not calendar days --
# documented explicitly, not glossed over).
nz = cedi[cedi != 0].to_numpy().reshape(-1, 1)
best_hmm, best_ll = None, -np.inf
for seed in range(10):
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=500,
                      random_state=seed, tol=1e-6)
    hmm.fit(nz)
    ll = hmm.score(nz)
    if ll > best_ll:
        best_ll, best_hmm = ll, hmm
hmm = best_hmm

if hmm is None:
    raise RuntimeError("All HMM initializations failed.")
print(f"Selected HMM log-likelihood: {best_ll:.6f} | "
      f"converged={hmm.monitor_.converged} | iterations={hmm.monitor_.iter}")
if not hmm.monitor_.converged:
    raise RuntimeError("Selected HMM initialization did not converge; "
                       "Treatment G should not be used.")

means = hmm.means_.ravel()
stds = np.sqrt(hmm.covars_.ravel())
quiet, volatile = (0, 1) if stds[0] < stds[1] else (1, 0)
print("2-state Markov-switching fit to nonzero cedi returns:")
print(f"  Quiet regime:    mean={means[quiet]:.4f}  std={stds[quiet]:.4f}")
print(f"  Volatile regime: mean={means[volatile]:.4f}  std={stds[volatile]:.4f}")
print(f"  Transition matrix:\n{np.round(hmm.transmat_, 4)}")
p_stay_quiet = hmm.transmat_[quiet, quiet]
p_stay_volatile = hmm.transmat_[volatile, volatile]
print(f"  Expected duration: quiet regime {1/(1-p_stay_quiet):.1f} obs, "
      f"volatile regime {1/(1-p_stay_volatile):.1f} obs")
print(f"  Stationary distribution: {np.round(hmm.get_stationary_distribution(), 4)}")

# ------------------------------------------------ forward algorithm: ONE-STEP-AHEAD
# predictive regime probabilities (information through t-1 only -- using
# r_t itself to weight the distribution r_t is evaluated against would be
# circular, exactly as GARCH's sigma_t never uses epsilon_t).
from scipy.special import logsumexp

T = len(nz)

log_emit = np.column_stack([
    stats.norm.logpdf(nz[:, 0], loc=means[j], scale=stds[j])
    for j in range(2)
])

alpha = np.zeros((T, 2))
pi_pred = np.zeros((T, 2))
pi_pred[0] = hmm.startprob_

logw = np.log(np.clip(pi_pred[0], 1e-300, None)) + log_emit[0]
alpha[0] = np.exp(logw - logsumexp(logw))

for t in range(1, T):
    pi_pred[t] = alpha[t - 1] @ hmm.transmat_
    logw = np.log(np.clip(pi_pred[t], 1e-300, None)) + log_emit[t]
    alpha[t] = np.exp(logw - logsumexp(logw))

print(f"\nOne-step-ahead P(quiet regime) stats: "
      f"mean={pi_pred[:, quiet].mean():.3f}, min={pi_pred[:, quiet].min():.3f}, "
      f"max={pi_pred[:, quiet].max():.3f}")

nz_index = cedi[cedi != 0].index


def hmm_mixture_cdf(x_val: float, pi_row: np.ndarray) -> float:
    return float(pi_row[quiet] * stats.norm.cdf(x_val, means[quiet], stds[quiet])
                + pi_row[volatile] * stats.norm.cdf(x_val, means[volatile], stds[volatile]))


F0_nz = np.array([hmm_mixture_cdf(0.0, pi_pred[i]) for i in range(T)])
Fz_nz = np.array([hmm_mixture_cdf(nz[i, 0], pi_pred[i]) for i in range(T)])
F0_series = pd.Series(F0_nz, index=nz_index)
Fz_series = pd.Series(Fz_nz, index=nz_index)

F0_stationary = hmm_mixture_cdf(0.0, hmm.get_stationary_distribution())

# ------------------------------------------------ Part 3: combine into full PIT
common_idx = p_t.index.intersection(cedi.index)
p_t2 = p_t.loc[common_idx]
r = cedi.loc[common_idx]
F0_full = pd.Series(F0_stationary, index=common_idx)
F0_full.loc[F0_full.index.intersection(F0_series.index)] = F0_series.loc[
    F0_full.index.intersection(F0_series.index)]

rng = np.random.default_rng(7)
u_mix = np.empty(len(r))
is_zero = (r.to_numpy() == 0)
is_pos = (r.to_numpy() > 0)
F0v = F0_full.loc[common_idx].to_numpy()
u_mix[is_zero] = ((1 - p_t2.to_numpy()[is_zero]) * F0v[is_zero]
                  + p_t2.to_numpy()[is_zero] * rng.uniform(0, 1, is_zero.sum()))

Fz_full = pd.Series(np.nan, index=common_idx)
Fz_full.loc[Fz_full.index.intersection(Fz_series.index)] = Fz_series.loc[
    Fz_full.index.intersection(Fz_series.index)]
Fzv = Fz_full.to_numpy()
u_mix[~is_zero & is_pos] = ((1 - p_t2.to_numpy()[~is_zero & is_pos]) * Fzv[~is_zero & is_pos]
                            + p_t2.to_numpy()[~is_zero & is_pos])
mask_neg = ~is_zero & ~is_pos
u_mix[mask_neg] = (1 - p_t2.to_numpy()[mask_neg]) * Fzv[mask_neg]

u_mix = np.clip(u_mix, 1e-9, 1 - 1e-9)
cedi_G = pd.Series(u_mix, index=common_idx, name="cedi")

print(f"\nu_mix histogram (10 bins):")
counts, edges = np.histogram(u_mix, bins=10, range=(0, 1))
for c, e in zip(counts, edges):
    print(f"  [{e:.1f}, {e+0.1:.1f}): {c}  (expected ~{len(u_mix)/10:.0f} if uniform)")

_pit_canonical = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
class _FitStub:
    def __init__(self, pit): self.pit = pit
fits = {c: _FitStub(_pit_canonical[c]) for c in _pit_canonical.columns}
ks_A = stats.kstest(fits["cedi"].pit.dropna(), "uniform")
ks_G = stats.kstest(cedi_G, "uniform")
print(f"\nKS test of PIT uniformity:")
print(f"  Treatment A (naive continuous, single-t):          p={ks_A.pvalue:.2e}")
print(f"  Treatment G (hurdle + Markov-switching, 2-state):  p={ks_G.pvalue:.4f}")

cedi_G.to_frame().to_csv("outputs/tables/cedi_hurdle_markov_pit.csv")