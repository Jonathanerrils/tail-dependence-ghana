"""Unified A-G cedi robustness diagnostic.

Scripts 10-13 construct seven cedi treatment specifications: the
canonical baseline (Treatment A) and six alternatives (Treatments B-G).
This script evaluates all seven using a common set of commodity-Cedi
dependence calculations for the combined COVID and 2024 stress
definition at q=0.025, 0.05, and 0.10.

The resulting table provides a common diagnostic comparison of
combined-stress lower-tail results across Treatments A-G. Full-sample
upper-tail concentration is also reported descriptively. This analysis
does not by itself establish robustness across the paper's separate
COVID-only and 2024-only stress definitions or across both tails; those
questions are addressed in the publication-resolution stress analysis.
"""

import sys
import hashlib

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model

from tailrisk import copulas, inference, marginals


def stable_seed(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


LABELS = {
    "cocoa": "Cocoa",
    "gold": "Gold",
    "brent": "Brent",
    "wti": "WTI",
}

COVID = ("2020-03-01", "2020-06-30")
C24 = ("2024-01-01", "2024-12-31")
QUANTILES = [0.025, 0.05, 0.10]


def stress_flag(idx):
    f = pd.Series(False, index=idx)
    for s, e in (COVID, C24):
        f |= (idx >= s) & (idx <= e)
    return f


def jitter_rank(x, rng):
    vals = x.to_numpy().astype(float)
    jitter = rng.uniform(-0.5, 0.5, size=len(vals))
    distinct = np.sort(np.unique(vals))
    min_gap = np.min(np.diff(distinct)) if len(distinct) > 1 else 1.0
    jittered = vals + jitter * min_gap * 0.99
    ranks = stats.rankdata(jittered)
    return pd.Series(ranks / (len(ranks) + 1), index=x.index)


pit_canonical = pd.read_csv(
    "outputs/tables/_pit_real.csv",
    index_col=0,
    parse_dates=True,
)

required_series = {"cocoa", "gold", "brent", "wti", "cedi"}
assert required_series.issubset(pit_canonical.columns), (
    "Canonical PIT file is missing required series: "
    f"{required_series - set(pit_canonical.columns)}"
)

rets = pd.read_csv(
    "data/processed/returns_real.csv",
    index_col=0,
    parse_dates=True,
)
cedi = rets["cedi"]

# ---- Treatments A-C: same constructions as script 10 ----
cedi_A = pit_canonical["cedi"]

cedi_B = jitter_rank(
    cedi.dropna(),
    np.random.default_rng(stable_seed("B_no_GARCH_raw_ranks")),
)

am = arch_model(
    cedi.dropna(),
    mean="AR",
    lags=1,
    vol="Constant",
    dist="t",
)
ar_fit = am.fit(disp="off")
assert ar_fit.convergence_flag == 0, (
    "Treatment C AR(1)-Student-t fit did not converge: "
    f"{ar_fit.convergence_flag}"
)
ar_resid = ar_fit.resid.dropna()

cedi_C = jitter_rank(
    ar_resid,
    np.random.default_rng(stable_seed("C_AR_only")),
)

# ---- Treatment D: weekly, full 5-series refit, cedi sign-corrected ----
prices = pd.read_csv(
    "data/processed/prices_real.csv",
    index_col=0,
    parse_dates=True,
)
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

assert required_series.issubset(pit_weekly.columns), (
    "Weekly PIT fit is missing required series: "
    f"{required_series - set(pit_weekly.columns)}"
)

# ---- Treatments E-G: PIT files exported by scripts 11-13 ----
cedi_E = pd.read_csv(
    "outputs/tables/cedi_hurdle_pit.csv",
    index_col=0,
    parse_dates=True,
)["cedi"]
cedi_F = pd.read_csv(
    "outputs/tables/cedi_hurdle_mixture_pit.csv",
    index_col=0,
    parse_dates=True,
)["cedi"]
cedi_G = pd.read_csv(
    "outputs/tables/cedi_hurdle_markov_pit.csv",
    index_col=0,
    parse_dates=True,
)["cedi"]

commodity_cols = ["cocoa", "gold", "brent", "wti"]

treatments = {
    "A_GARCH_baseline": (pit_canonical[commodity_cols], cedi_A),
    "B_no_GARCH_raw_ranks": (pit_canonical[commodity_cols], cedi_B),
    "C_AR_only": (pit_canonical[commodity_cols], cedi_C),
    "D_weekly_frequency": (pit_weekly[commodity_cols], pit_weekly["cedi"]),
    "E_hurdle_studentt": (pit_canonical[commodity_cols], cedi_E),
    "F_hurdle_mixture": (pit_canonical[commodity_cols], cedi_F),
    "G_hurdle_markov": (pit_canonical[commodity_cols], cedi_G),
}

results = []
for tname, (comm_pit, cedi_alt) in treatments.items():
    common_idx = comm_pit.index.intersection(cedi_alt.index)
    comm = comm_pit.loc[common_idx]
    cedi_series = cedi_alt.loc[common_idx]
    flag = stress_flag(common_idx)

    for c in commodity_cols:
        u_all = copulas.pseudo_obs(comm[[c]])[:, 0]
        v_all = stats.rankdata(cedi_series.to_numpy()) / (len(cedi_series) + 1)

        for q in QUANTILES:
            lam_L = copulas.empirical_lambda_lower(u_all, v_all, q)
            lam_U = copulas.empirical_lambda_upper(u_all, v_all, 1 - q)

            calm_mask = ~flag.to_numpy()
            stress_mask = flag.to_numpy()
            u_c, v_c = u_all[calm_mask], v_all[calm_mask]
            u_s, v_s = u_all[stress_mask], v_all[stress_mask]

            u_c = stats.rankdata(u_c) / (len(u_c) + 1)
            v_c = stats.rankdata(v_c) / (len(v_c) + 1)
            u_s = stats.rankdata(u_s) / (len(u_s) + 1)
            v_s = stats.rankdata(v_s) / (len(v_s) + 1)

            r = inference.calm_stress_difference(
                u_c,
                v_c,
                u_s,
                v_s,
                q=q,
                n_boot=500,
                seed=stable_seed(tname, c, q),
            )

            results.append({
                "treatment": tname,
                "pair": f"{LABELS[c]}\u2013Cedi",
                "q": q,
                "n": len(common_idx),
                "lambda_L_full": lam_L,
                "lambda_U_full": lam_U,
                "lambda_calm": r["lambda_calm"],
                "lambda_stress": r["lambda_stress"],
                "difference": r["difference"],
                "p_value": r["p_value"],
                "sig_nominal_5pct": r["significant_5pct"],
            })

    print(f"done: {tname} (n={len(common_idx)})")

df = pd.DataFrame(results)
df.round(4).to_csv(
    "outputs/tables/cedi_robustness_AG_full.csv",
    index=False,
)

print(
    "\n=== Nominal 5% significance count by treatment "
    "(12 diagnostic tests each; no multiplicity correction) ==="
)
print(df.groupby("treatment")["sig_nominal_5pct"].agg(["sum", "count"]))

print("\n=== Full table ===")
print(df.round(4).to_string(index=False))

# ------------------------------------------------ validation
expected_treatments = {
    "A_GARCH_baseline",
    "B_no_GARCH_raw_ranks",
    "C_AR_only",
    "D_weekly_frequency",
    "E_hurdle_studentt",
    "F_hurdle_mixture",
    "G_hurdle_markov",
}
expected_pairs = {
    "Cocoa\u2013Cedi",
    "Gold\u2013Cedi",
    "Brent\u2013Cedi",
    "WTI\u2013Cedi",
}
expected_q = {0.025, 0.05, 0.10}

assert len(df) == 84, f"Expected 84 rows, got {len(df)}"
assert set(df["treatment"]) == expected_treatments
assert set(df["pair"]) == expected_pairs
assert set(df["q"].round(3)) == expected_q

assert not df.duplicated(subset=["treatment", "pair", "q"]).any(), \
    "Duplicate treatment/pair/q rows found"

required_numeric = [
    "n",
    "lambda_L_full",
    "lambda_U_full",
    "lambda_calm",
    "lambda_stress",
    "difference",
    "p_value",
]

assert not df[required_numeric].isna().any().any(), \
    "Missing values found in robustness output"

assert df["sig_nominal_5pct"].notna().all(), \
    "Missing nominal significance flags found"

print("\n=== STRUCTURAL VALIDATION PASSED ===")
print("Rows:       ", len(df))
print("Treatments: ", df["treatment"].nunique())
print("Pairs:      ", df["pair"].nunique())
print("Quantiles:  ", sorted(df["q"].unique()))
print("Duplicates: ", df.duplicated(subset=["treatment", "pair", "q"]).sum())
