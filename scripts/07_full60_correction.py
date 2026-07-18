import sys, itertools
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
flag = pd.Series(False, index=idx)
for s, e in (COVID, C24):
    flag |= (idx >= s) & (idx <= e)
pit_calm, pit_stress = pit[~flag], pit[flag]
print(f"Calm n={len(pit_calm)}, Stress n={len(pit_stress)} "
      f"(COVID {(( idx>=COVID[0])&(idx<=COVID[1])).sum()} + "
      f"2024 {((idx>=C24[0])&(idx<=C24[1])).sum()})")

M = 60  # full family-wise: 10 pairs x 3 quantiles x 2 tails
rows = []
for q in (0.025, 0.05, 0.10):
    for tail, upper in (("lower", False), ("upper", True)):
        for a, b in pairs:
            u_c = copulas.pseudo_obs(pit_calm[[a]])[:, 0]
            v_c = copulas.pseudo_obs(pit_calm[[b]])[:, 0]
            u_s = copulas.pseudo_obs(pit_stress[[a]])[:, 0]
            v_s = copulas.pseudo_obs(pit_stress[[b]])[:, 0]
            r = inference.calm_stress_difference(
                u_c, v_c, u_s, v_s, q=q, n_boot=500, level=0.95,
                seed=hash((a, b, q, tail)) % (2**31), bonf_m=M, upper=upper)
            rows.append({
                "q": q, "tail": tail, "pair": f"{LABELS[a]}\u2013{LABELS[b]}",
                "lambda_calm": r["lambda_calm"], "lambda_stress": r["lambda_stress"],
                "difference": r["difference"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                "p_value": r["p_value"],
                "sig_uncorrected_5pct": r["significant_5pct"],
                "diff_lo_bonf60": r["diff_lo_bonf"], "diff_hi_bonf60": r["diff_hi_bonf"],
                "sig_bonferroni_m60": r["significant_bonferroni"],
            })
    print(f"q={q} done ({len(rows)} rows so far)", flush=True)

df = pd.DataFrame(rows)

# Benjamini-Hochberg FDR across all 60 p-values simultaneously
p = df["p_value"].to_numpy()
order = np.argsort(p)
ranked = p[order]
m = len(p)
thresh = (np.arange(1, m + 1) / m) * 0.05
below = ranked <= thresh
if below.any():
    k_max = np.max(np.where(below)[0])
    bh_cutoff_p = ranked[k_max]
else:
    bh_cutoff_p = -1.0
df["sig_FDR_BH_5pct"] = df["p_value"] <= bh_cutoff_p
df["bh_cutoff_p_used"] = bh_cutoff_p

df.round(5).to_csv("outputs/tables/calm_stress_full60_correction.csv", index=False)

print(f"\nBH-FDR cutoff p-value used: {bh_cutoff_p:.5f}")
print(f"\nSignificant cells:")
print(f"  Uncorrected (5%):        {int(df['sig_uncorrected_5pct'].sum())} / 60")
print(f"  Bonferroni (m=60):       {int(df['sig_bonferroni_m60'].sum())} / 60")
print(f"  Benjamini-Hochberg FDR:  {int(df['sig_FDR_BH_5pct'].sum())} / 60")

print("\n--- Cells significant under ANY correction ---")
any_sig = df[df["sig_uncorrected_5pct"]]
print(any_sig[["q", "tail", "pair", "difference", "p_value",
               "sig_uncorrected_5pct", "sig_bonferroni_m60", "sig_FDR_BH_5pct"]]
      .sort_values("p_value").to_string(index=False))
