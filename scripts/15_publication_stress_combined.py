"""Publication-resolution combined-stress test: 15,000 bootstrap
replicates, Bonferroni and BH decisions computed directly from the
centered bootstrap p-value (not the percentile CI). Writes to a new
filename -- the 500-draw diagnostic output is kept as a separate audit
trail, not overwritten.
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
M = 60  # full family-wise: 10 pairs x 3 quantiles x 2 tails

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
      f"(COVID {((idx>=COVID[0])&(idx<=COVID[1])).sum()} + "
      f"2024 {((idx>=C24[0])&(idx<=C24[1])).sum()})")
print(f"N_BOOT={N_BOOT}, M={M}, Bonferroni threshold=0.05/{M}={0.05/M:.6f}")

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
                seed=stable_seed("combined", a, b, q, tail), upper=upper)
            rows.append({
                "q": q, "tail": tail, "pair": f"{LABELS[a]}\u2013{LABELS[b]}",
                "lambda_calm": r["lambda_calm"], "lambda_stress": r["lambda_stress"],
                "difference": r["difference"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                "p_value": r["p_value"],
                "sig_uncorrected_5pct": r["significant_5pct"],
            })
    print(f"q={q} done ({len(rows)} rows so far, {time.time()-t0:.1f}s elapsed)", flush=True)

df = pd.DataFrame(rows)
df["sig_bonferroni_m60"] = df["p_value"] <= 0.05 / M

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

# Structural assertions before writing anything
assert len(df) == 60, f"Expected 60 rows, got {len(df)}"
assert df["pair"].nunique() == 10, f"Expected 10 unique pairs, got {df['pair'].nunique()}"
assert set(df["q"]) == {0.025, 0.05, 0.10}, f"Unexpected q values: {set(df['q'])}"
assert set(df["tail"]) == {"lower", "upper"}, f"Unexpected tail values: {set(df['tail'])}"
assert not df.duplicated(["q", "tail", "pair"]).any(), "Duplicate q/tail/pair rows found"
assert df["p_value"].between(0, 1).all(), "p-values outside [0,1] found"
assert not df.isna().any().any(), "Missing values found in output"

# Platform-independent serialization: sorted row order, fixed float
# precision, explicit line terminator and encoding, so a matching
# SHA-256 across Windows and Linux actually means something.
df.sort_values(["q", "tail", "pair"]).to_csv(
    "outputs/tables/calm_stress_full60_B15000.csv",
    index=False,
    float_format="%.6f",
    lineterminator="\n",
    encoding="utf-8",
)

print(f"\nTotal time: {time.time()-t0:.1f}s")
print(f"BH-FDR cutoff p-value used: {bh_cutoff_p:.6f}")
print(f"Minimum p-value observed: {p.min():.6f}")
print(f"\nSignificant cells:")
print(f"  Uncorrected (5%):        {int(df['sig_uncorrected_5pct'].sum())} / 60")
print(f"  Bonferroni (m=60):       {int(df['sig_bonferroni_m60'].sum())} / 60")
print(f"  Benjamini-Hochberg FDR:  {int(df['sig_FDR_BH_5pct'].sum())} / 60")

print("\n--- Cells significant under ANY correction ---")
any_sig = df[df["sig_uncorrected_5pct"]]
print(any_sig[["q", "tail", "pair", "difference", "p_value",
               "sig_uncorrected_5pct", "sig_bonferroni_m60", "sig_FDR_BH_5pct"]]
      .sort_values("p_value").to_string(index=False))

print("\nStructural assertions: PASSED")