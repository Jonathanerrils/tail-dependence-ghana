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
# (unchanged: validated earlier, zeros genuinely cluster)
I_t = (cedi == 0).astype(int)
I_lag1, I_lag2 = I_t.shift(1), I_t.shift(2)
df_logit = pd.DataFrame({"I": I_t, "I1": I_lag1, "I2": I_lag2}).dropna()
logit2 = sm.Logit(df_logit["I"], sm.add_constant(df_logit[["I1", "I2"]])).fit(disp=0)
X = sm.add_constant(pd.DataFrame({"I1": I_lag1, "I2": I_lag2}))
p_t = logit2.predict(X.dropna())

# ------------------------------------------------ Part 2: 2-state Markov-switching
# continuous component, fit on the nonzero subsequence (same compression
# simplification as the static-mixture treatment: transition probabilities
# apply between consecutive NONZERO observations, not calendar days --
# documented explicitly, not glossed over).
nz = cedi[cedi != 0].to_numpy().reshape(-1, 1)
best_hmm, best_bic = None, np.inf
for seed in range(10):
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=500,
                      random_state=seed, tol=1e-6)
    hmm.fit(nz)
    ll = hmm.score(nz)
    n_params = 2 + 2 * 2 + 2 * 2 - 1  # startprob(1 free)+transmat(2 free)+means(2)+vars(2), rough count
    bic = -2 * ll + n_params * np.log(len(nz))
    if bic < best_bic:
        best_bic, best_hmm = bic, hmm
hmm = best_hmm

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
T = len(nz)
log_emit = hmm._compute_log_likelihood(nz)  # T x 2, log density of r_t under each state
emit = np.exp(log_emit - log_emit.max(axis=1, keepdims=True))  # stabilized
emit = emit / emit.sum(axis=1, keepdims=True) * np.exp(log_emit.max(axis=1, keepdims=True))
alpha = np.zeros((T, 2))
pi_pred = np.zeros((T, 2))  # one-step-ahead predicted state probs, used for the PIT
pi_pred[0] = hmm.get_stationary_distribution()
alpha[0] = pi_pred[0] * np.exp(log_emit[0])
alpha[0] /= alpha[0].sum()
for t in range(1, T):
    pi_pred[t] = alpha[t - 1] @ hmm.transmat_
    alpha[t] = pi_pred[t] * np.exp(log_emit[t])
    alpha[t] /= alpha[t].sum()

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

# For zero-return days (not in the nonzero HMM), use the stationary
# distribution as the regime weight (no better information available
# for a calendar day that fell in the excluded/hurdle-modeled subset).
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

with open("outputs/fits.pkl", "rb") as f:
    fits = pickle.load(f)
ks_A = stats.kstest(fits["cedi"].pit.dropna(), "uniform")
ks_G = stats.kstest(cedi_G, "uniform")
print(f"\nKS test of PIT uniformity:")
print(f"  Treatment A (naive continuous, single-t):          p={ks_A.pvalue:.2e}")
print(f"  Treatment G (hurdle + Markov-switching, 2-state):  p={ks_G.pvalue:.4f}")

cedi_G.to_frame().to_csv("outputs/tables/cedi_hurdle_markov_pit.csv")
