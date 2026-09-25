import sys, hashlib
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from tailrisk import copulas, inference, marginals

def stable_seed(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

rets = pd.read_csv("data/processed/returns_real.csv", index_col=0, parse_dates=True)
cedi = rets["cedi"]
LABELS = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI"}

rng = np.random.default_rng(42)


def jitter_rank(x: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Continuization (Denuit & Lambert 2005 style): within each group of
    tied values (dominated here by the cedi's exact-zero mass), add small
    uniform jitter before ranking, so tied observations are broken
    randomly rather than all mapped to the same pseudo-observation. This
    is the standard fix for discrete/atomic marginals in rank-based
    copula estimation, rather than pretending a continuous density fits."""
    vals = x.to_numpy().astype(float)
    jitter = rng.uniform(-0.5, 0.5, size=len(vals))
    distinct = np.sort(np.unique(vals))
    min_gap = np.min(np.diff(distinct)) if len(distinct) > 1 else 1.0
    jittered = vals + jitter * min_gap * 0.99
    ranks = stats.rankdata(jittered)
    return pd.Series(ranks / (len(ranks) + 1), index=x.index)


pit_baseline = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
cedi_A = pit_baseline["cedi"]

# ---------------------------------------------------- Treatment B: no-GARCH raw ranks
cedi_B_full = jitter_rank(cedi.dropna(), rng)

# ---------------------------------------------------- Treatment C: AR(1)-only mean filter (no GARCH variance filter)
am = arch_model(cedi.dropna(), mean="AR", lags=1, vol="Constant", dist="t")
res = am.fit(disp="off")
ar_resid = res.resid.dropna()
cedi_C_full = jitter_rank(ar_resid, rng)

# ---------------------------------------------------- Treatment D: weekly frequency
prices = pd.read_csv("data/processed/prices_real.csv", index_col=0, parse_dates=True)
weekly_px = prices.resample("W-FRI").last()
weekly_ret = 100 * np.log(weekly_px).diff().dropna()
weekly_ret["cedi"] = -weekly_ret["cedi"]

weekly_raw_cedi = 100 * np.log(weekly_px["cedi"]).diff().dropna()
_common = weekly_ret.index.intersection(weekly_raw_cedi.index)
assert np.allclose(
    weekly_ret.loc[_common, "cedi"].to_numpy(),
    -weekly_raw_cedi.loc[_common].to_numpy(),
    equal_nan=False,
), "Weekly cedi orientation is wrong"
wfits, _ = marginals.fit_all(weekly_ret, gate=True)
pit_weekly = pd.concat([f.pit for f in wfits.values()], axis=1).dropna()

print("Sample sizes: baseline(GARCH)=", len(pit_baseline), "| no-GARCH raw=", len(cedi_B_full),
      "| AR-only=", len(cedi_C_full), "| weekly=", len(pit_weekly))

results = []
COVID, C24 = ("2020-03-01", "2020-06-30"), ("2024-01-01", "2024-12-31")

def stress_flag(idx):
    f = pd.Series(False, index=idx)
    for s, e in (COVID, C24):
        f |= (idx >= s) & (idx <= e)
    return f

treatments = {
    "A_GARCH_baseline": (pit_baseline[["cocoa", "gold", "brent", "wti"]], cedi_A),
    "B_no_GARCH_raw_ranks": (pit_baseline[["cocoa", "gold", "brent", "wti"]].loc[cedi_B_full.index.intersection(pit_baseline.index)],
                             cedi_B_full),
    "C_AR_only": (pit_baseline[["cocoa", "gold", "brent", "wti"]].loc[cedi_C_full.index.intersection(pit_baseline.index)],
                    cedi_C_full),
}

for tname, (comm_pit, cedi_alt) in treatments.items():
    common_idx = comm_pit.index.intersection(cedi_alt.index)
    comm = comm_pit.loc[common_idx]
    cedi_series = cedi_alt.loc[common_idx]
    flag = stress_flag(common_idx)
    for c in ["cocoa", "gold", "brent", "wti"]:
        u_all = copulas.pseudo_obs(comm[[c]])[:, 0]
        v_all = cedi_series.to_numpy()
        v_all = stats.rankdata(v_all) / (len(v_all) + 1)
        lam_L = copulas.empirical_lambda_lower(u_all, v_all, 0.05)
        lam_U = copulas.empirical_lambda_upper(u_all, v_all, 0.95)
        u_c, v_c = u_all[~flag.to_numpy()], v_all[~flag.to_numpy()]
        u_s, v_s = u_all[flag.to_numpy()], v_all[flag.to_numpy()]
        u_c = stats.rankdata(u_c) / (len(u_c) + 1); v_c = stats.rankdata(v_c) / (len(v_c) + 1)
        u_s = stats.rankdata(u_s) / (len(u_s) + 1); v_s = stats.rankdata(v_s) / (len(v_s) + 1)
        r = inference.calm_stress_difference(u_c, v_c, u_s, v_s, q=0.05, n_boot=500,
                                             seed=stable_seed(tname, c))
        results.append({"treatment": tname, "pair": f"{LABELS[c]}\u2013Cedi",
                        "n": len(common_idx), "lambda_L_full": lam_L, "lambda_U_full": lam_U,
                        "lambda_calm": r["lambda_calm"], "lambda_stress": r["lambda_stress"],
                        "difference": r["difference"], "p_value": r["p_value"],
                        "sig_5pct": r["significant_5pct"]})

widx = pit_weekly.index
wflag = stress_flag(widx)
for c in ["cocoa", "gold", "brent", "wti"]:
    u_all = copulas.pseudo_obs(pit_weekly[[c]])[:, 0]
    v_all = copulas.pseudo_obs(pit_weekly[["cedi"]])[:, 0]
    lam_L = copulas.empirical_lambda_lower(u_all, v_all, 0.05)
    lam_U = copulas.empirical_lambda_upper(u_all, v_all, 0.95)
    u_c = copulas.pseudo_obs(pit_weekly.loc[~wflag, [c]])[:, 0]
    v_c = copulas.pseudo_obs(pit_weekly.loc[~wflag, ["cedi"]])[:, 0]
    u_s = copulas.pseudo_obs(pit_weekly.loc[wflag, [c]])[:, 0]
    v_s = copulas.pseudo_obs(pit_weekly.loc[wflag, ["cedi"]])[:, 0]
    r = inference.calm_stress_difference(u_c, v_c, u_s, v_s, q=0.05, n_boot=500,
                                         seed=stable_seed("D", c))
    results.append({"treatment": "D_weekly_frequency", "pair": f"{LABELS[c]}\u2013Cedi",
                    "n": len(widx), "lambda_L_full": lam_L, "lambda_U_full": lam_U,
                    "lambda_calm": r["lambda_calm"], "lambda_stress": r["lambda_stress"],
                    "difference": r["difference"], "p_value": r["p_value"],
                    "sig_5pct": r["significant_5pct"]})

df = pd.DataFrame(results)
df.round(4).to_csv("outputs/tables/cedi_alternative_treatments.csv", index=False)
print("\n" + df.round(4).to_string(index=False))