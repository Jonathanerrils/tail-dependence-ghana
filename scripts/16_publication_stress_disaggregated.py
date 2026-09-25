"""Publication-resolution COVID-only and 2024-only stress tests, 15,000
bootstrap replicates each, same p-value-based decision rule as the
combined-stress run.
"""

import sys, itertools, hashlib, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from tailrisk import copulas, inference

def stable_seed(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

N_BOOT = 15_000
M = 60

pit = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
idx = pit.index
LABELS = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI", "cedi": "Cedi"}
pairs = list(itertools.combinations(pit.columns, 2))

COVID = ("2020-03-01", "2020-06-30")
C24 = ("2024-01-01", "2024-12-31")
covid_flag = (idx >= COVID[0]) & (idx <= COVID[1])
c24_flag = (idx >= C24[0]) & (idx <= C24[1])
either_flag = covid_flag | c24_flag

pit_calm = pit[~either_flag]
pit_covid = pit[covid_flag]
pit_2024 = pit[c24_flag]
print(f"Calm (excl. both): n={len(pit_calm)} | COVID-only: n={len(pit_covid)} | "
      f"2024-only: n={len(pit_2024)}")

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
                    u_c, v_c, u_s, v_s, q=q, n_boot=N_BOOT, level=0.95,
                    seed=stable_seed(tag, a, b, q, tail), upper=upper)
                rows.append({
                    "stress_def": tag, "q": q, "tail": tail,
                    "pair": f"{LABELS[a]}\u2013{LABELS[b]}",
                    "lambda_calm": r["lambda_calm"], "lambda_stress": r["lambda_stress"],
                    "difference": r["difference"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                    "p_value": r["p_value"],
                    "sig_uncorrected_5pct": r["significant_5pct"],
                })
        print(f"  [{tag}] q={q} done ({len(rows)} rows, {time.time()-t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["sig_bonferroni_m60"] = df["p_value"] <= 0.05 / M
    p = df["p_value"].to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresh = (np.arange(1, m + 1) / m) * 0.05
    below = ranked <= thresh
    bh_cutoff_p = ranked[np.max(np.where(below)[0])] if below.any() else -1.0
    df["sig_FDR_BH_5pct"] = df["p_value"] <= bh_cutoff_p
    print(f"[{tag}] {time.time()-t0:.1f}s | BH cutoff p={bh_cutoff_p:.6f} | "
          f"min p={p.min():.6f} | "
          f"uncorrected={int(df['sig_uncorrected_5pct'].sum())}/60  "
          f"bonf60={int(df['sig_bonferroni_m60'].sum())}/60  "
          f"FDR={int(df['sig_FDR_BH_5pct'].sum())}/60")
    return df

df_covid = run_sweep(pit_covid, "COVID_only")
df_2024 = run_sweep(pit_2024, "2024_only")

full = pd.concat([df_covid, df_2024], ignore_index=True)

# Structural assertions before writing anything
assert len(full) == 120, f"Expected 120 rows, got {len(full)}"
assert set(full["stress_def"]) == {"COVID_only", "2024_only"}
assert not full.duplicated(["stress_def", "q", "tail", "pair"]).any(), \
    "Duplicate stress_def/q/tail/pair rows found"
assert full["p_value"].between(0, 1).all(), "p-values outside [0,1] found"
assert not full.isna().any().any(), "Missing values found in output"

full.sort_values(["stress_def", "q", "tail", "pair"]).to_csv(
    "outputs/tables/calm_stress_disaggregated_B15000.csv",
    index=False,
    float_format="%.6f",
    lineterminator="\n",
    encoding="utf-8",
)

print("\n=== COVID-only: cells significant uncorrected ===")
print(df_covid[df_covid["sig_uncorrected_5pct"]]
      [["q", "tail", "pair", "difference", "p_value", "sig_bonferroni_m60", "sig_FDR_BH_5pct"]]
      .sort_values("p_value").to_string(index=False))

print("\n=== 2024-only: cells significant uncorrected ===")
print(df_2024[df_2024["sig_uncorrected_5pct"]]
      [["q", "tail", "pair", "difference", "p_value", "sig_bonferroni_m60", "sig_FDR_BH_5pct"]]
      .sort_values("p_value").to_string(index=False))

print("\nStructural assertions: PASSED")