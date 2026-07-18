import sys, itertools, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from tailrisk import copulas, inference

pit = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
idx = pit.index
LABELS = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI", "cedi": "Cedi"}
pairs = list(itertools.combinations(pit.columns, 2))

COVID = ("2020-03-01", "2020-06-30")
C24 = ("2024-01-01", "2024-12-31")
covid_flag = (idx >= COVID[0]) & (idx <= COVID[1])
c24_flag = (idx >= C24[0]) & (idx <= C24[1])
either_flag = covid_flag | c24_flag

# Calm baseline held CONSTANT across both disaggregated analyses: excludes
# BOTH crisis windows, exactly as in the combined-stress (primary) analysis,
# so a COVID-only vs 2024-only difference reflects which window drives any
# signal, not a shifting comparison group.
pit_calm = pit[~either_flag]
pit_covid = pit[covid_flag]
pit_2024 = pit[c24_flag]
print(f"Calm (excl. both): n={len(pit_calm)} | COVID-only: n={len(pit_covid)} | "
      f"2024-only: n={len(pit_2024)}")

M = 60

def run_sweep(pit_stress, tag):
    rows = []
    t0 = time.time()
    for q in (0.025, 0.05, 0.10):
        for tail, upper in (("lower", False), ("upper", True)):
            for a, b in pairs:
                u_c = copulas.pseudo_obs(pit_calm[[a]])[:, 0]
                v_c = copulas.pseudo_obs(pit_calm[[b]])[:, 0]
                u_s = copulas.pseudo_obs(pit_stress[[a]])[:, 0]
                v_s = copulas.pseudo_obs(pit_stress[[b]])[:, 0]
                r = inference.calm_stress_difference(
                    u_c, v_c, u_s, v_s, q=q, n_boot=500, level=0.95,
                    seed=hash((tag, a, b, q, tail)) % (2**31), bonf_m=M, upper=upper)
                rows.append({
                    "stress_def": tag, "q": q, "tail": tail,
                    "pair": f"{LABELS[a]}\u2013{LABELS[b]}",
                    "lambda_calm": r["lambda_calm"], "lambda_stress": r["lambda_stress"],
                    "difference": r["difference"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                    "p_value": r["p_value"],
                    "sig_uncorrected_5pct": r["significant_5pct"],
                    "sig_bonferroni_m60": r["significant_bonferroni"],
                })
    df = pd.DataFrame(rows)
    p = df["p_value"].to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresh = (np.arange(1, m + 1) / m) * 0.05
    below = ranked <= thresh
    bh_cutoff_p = ranked[np.max(np.where(below)[0])] if below.any() else -1.0
    df["sig_FDR_BH_5pct"] = df["p_value"] <= bh_cutoff_p
    print(f"[{tag}] {time.time()-t0:.1f}s | BH cutoff p={bh_cutoff_p:.5f} | "
          f"uncorrected={int(df['sig_uncorrected_5pct'].sum())}/60  "
          f"bonf60={int(df['sig_bonferroni_m60'].sum())}/60  "
          f"FDR={int(df['sig_FDR_BH_5pct'].sum())}/60")
    return df

df_covid = run_sweep(pit_covid, "COVID_only")
df_2024 = run_sweep(pit_2024, "2024_only")

full = pd.concat([df_covid, df_2024], ignore_index=True)
full.round(5).to_csv("outputs/tables/calm_stress_disaggregated.csv", index=False)

print("\n=== COVID-only: cells significant uncorrected ===")
print(df_covid[df_covid["sig_uncorrected_5pct"]]
      [["q", "tail", "pair", "difference", "p_value", "sig_bonferroni_m60", "sig_FDR_BH_5pct"]]
      .sort_values("p_value").to_string(index=False))

print("\n=== 2024-only: cells significant uncorrected ===")
print(df_2024[df_2024["sig_uncorrected_5pct"]]
      [["q", "tail", "pair", "difference", "p_value", "sig_bonferroni_m60", "sig_FDR_BH_5pct"]]
      .sort_values("p_value").to_string(index=False))

# Mean lambda by q, each stress definition, both tails -- the headline
# summary table for the paper's disaggregated results section.
summary = full.groupby(["stress_def", "tail", "q"])[["lambda_calm", "lambda_stress"]].mean()
summary["change"] = summary["lambda_stress"] - summary["lambda_calm"]
summary.round(4).to_csv("outputs/tables/calm_stress_disaggregated_means.csv")
print("\n=== Mean lambda by stress definition, tail, q ===")
print(summary.round(4).to_string())
