import sys, itertools, time
sys.path.insert(0, "src")
import pandas as pd
from tailrisk import copulas, gof

pit = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0)
LABELS = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI", "cedi": "Cedi"}
pairs = list(itertools.combinations(pit.columns, 2))
FAMILIES = ["gaussian", "t", "clayton", "gumbel", "frank", "joe",
            "survival_clayton", "survival_gumbel"]

rows = []
t0 = time.time()
for i, (a, b) in enumerate(pairs):
    u, v = copulas.pseudo_obs(pit[[a]])[:, 0], copulas.pseudo_obs(pit[[b]])[:, 0]
    for fam in FAMILIES:
        r = gof.gof_test(u, v, fam, n_boot=500, seed=hash((a, b, fam)) % (2**31))
        rows.append({"pair": f"{LABELS[a]}\u2013{LABELS[b]}", "family": fam,
                     "Sn": r["Sn"], "p_value": r["p_value"],
                     "not_rejected_5pct": r["p_value"] >= 0.05, **r["params"]})
    print(f"[{time.time()-t0:6.1f}s] done {LABELS[a]}-{LABELS[b]} ({i+1}/{len(pairs)})",
          flush=True)

df = pd.DataFrame(rows)
df.round(4).to_csv("outputs/tables/copula_gof_8family.csv", index=False)
print(f"\nTotal time: {time.time()-t0:.1f}s")

summ = df.groupby("pair")["not_rejected_5pct"].sum().rename("n_not_rejected_of_8")
print(summ.to_string())
