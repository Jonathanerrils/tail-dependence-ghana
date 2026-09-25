"""Publication rebuild v2 — Stage 2: copula fits, empirical tails, GOF.

Non-destructive. Reads Stage 1 PITs and writes only to
outputs/publication_rebuild_v2/tables/.

Main GOF run: 80 cells (10 pairs x 8 families), B=2000.
Borderline cells with initial p in [0.03, 0.07] are rerun at B=5000.
Monte Carlo p-value = (exceed + 1)/(B + 1).
Stable SHA256-derived seeds and resumable checkpointing.

Run:
    python scripts/29_publication_rebuild_stage2_gof.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tailrisk import copulas, gof  # noqa: E402

V2 = ROOT / "outputs" / "publication_rebuild_v2" / "tables"
PITFILE = V2 / "_pit_publication_v2.csv"
CHECKPOINT = V2 / "gof_B2000_checkpoint.csv"
FINAL = V2 / "gof_publication_v2_final.csv"
FITS = V2 / "copula_fits_publication_v2.csv"
EMP = V2 / "empirical_tail_concentration_fullsample_v2.csv"
PAIRSUM = V2 / "gof_pair_summary_v2.csv"
BORDERFILE = V2 / "gof_borderline_B5000.csv"
SUMMARY = V2 / "gof_stage2_summary.json"
HASHES = V2 / "stage2_hashes.txt"

FAMILIES = [
    "gaussian", "t", "clayton", "gumbel",
    "frank", "joe", "survival_clayton", "survival_gumbel",
]
B_MAIN, B_BORDER, GRID = 2000, 5000, 50
BORDER_LO, BORDER_HI, ALPHA = 0.03, 0.07, 0.05
LABELS = {"cocoa":"Cocoa","gold":"Gold","brent":"Brent","wti":"WTI","cedi":"Cedi"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_seed(*parts) -> int:
    b = "||".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big") % (2**32 - 1)

def plabel(a, b):
    return f"{LABELS.get(a,a)}–{LABELS.get(b,b)}"

def pseudo_pair(frame, a, b):
    po = copulas.pseudo_obs(frame[[a, b]])
    return po[:, 0], po[:, 1]

def fit_family(fam, u, v):
    f = copulas.FAMILIES[fam](u, v)
    return f, f.params

def run_cell(u, v, fam, B, seed):
    rng = np.random.default_rng(seed)
    fit0, p0 = fit_family(fam, u, v)
    s0 = gof.cvm_rosenblatt(u, v, fam, p0, grid_size=GRID)
    exceed = 0
    for _ in range(B):
        su, sv = gof.sample(fam, p0, len(u), rng)
        su = copulas.pseudo_obs(np.asarray(su).reshape(-1, 1))[:, 0]
        sv = copulas.pseudo_obs(np.asarray(sv).reshape(-1, 1))[:, 0]
        _, pb = fit_family(fam, su, sv)
        sb = gof.cvm_rosenblatt(su, sv, fam, pb, grid_size=GRID)
        exceed += int(sb >= s0)
    return {
        "family": fam,
        "Sn": float(s0),
        "p_value": float((exceed + 1) / (B + 1)),
        "exceed": int(exceed),
        "n_boot": int(B),
        "params": json.dumps(p0, sort_keys=True),
        "loglik": float(fit0.loglik),
        "AIC": float(fit0.aic),
        "lambda_lower_model": float(fit0.lambda_lower),
        "lambda_upper_model": float(fit0.lambda_upper),
    }

def atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g")
    os.replace(tmp, path)

if not PITFILE.exists():
    raise FileNotFoundError(PITFILE)

pit = pd.read_csv(PITFILE, index_col=0, parse_dates=True).sort_index()
pairs = list(itertools.combinations(pit.columns, 2))
assert len(pairs) == 10

print("=== PUBLICATION REBUILD V2: STAGE 2 GOF ===")
print(f"PIT SHA256: {sha256(PITFILE)}")
print(f"shape: {pit.shape}; period: {pit.index.min().date()} to {pit.index.max().date()}")
print(f"cells: {len(pairs)*len(FAMILIES)}; B_main={B_MAIN}; B_border={B_BORDER}")
print()

cache = {}
fit_rows, emp_rows = [], []
for a, b in pairs:
    u, v = pseudo_pair(pit, a, b)
    cache[(a, b)] = (u, v)
    fitd = copulas.fit_pair(u, v)
    best = min(fitd.values(), key=lambda x: x.aic)
    for fam in FAMILIES:
        f = fitd[fam]
        fit_rows.append({
            "pair":plabel(a,b),"a":a,"b":b,"family":fam,"n":len(u),
            "loglik":f.loglik,"AIC":f.aic,
            "best_AIC_within_candidates":fam==best.family,
            "params":json.dumps(f.params,sort_keys=True),
            "lambda_lower_model":f.lambda_lower,
            "lambda_upper_model":f.lambda_upper,
        })
    for q in (0.025,0.05,0.10):
        emp_rows.append({"pair":plabel(a,b),"a":a,"b":b,"q":q,"tail":"lower",
                         "lambda_empirical":copulas.empirical_lambda_lower(u,v,q)})
        emp_rows.append({"pair":plabel(a,b),"a":a,"b":b,"q":q,"tail":"upper",
                         "lambda_empirical":copulas.empirical_lambda_upper(u,v,1-q)})

pd.DataFrame(fit_rows).to_csv(FITS,index=False,float_format="%.12g")
pd.DataFrame(emp_rows).to_csv(EMP,index=False,float_format="%.12g")

if CHECKPOINT.exists():
    cp = pd.read_csv(CHECKPOINT)
    rows = cp.to_dict("records")
    done = {(r["a"],r["b"],r["family"]) for _,r in cp.iterrows()
            if int(r["n_boot"]) == B_MAIN}
    print(f"Resuming checkpoint: {len(done)}/80 complete.")
else:
    rows, done = [], set()

t0 = time.time()
for a,b in pairs:
    u,v = cache[(a,b)]
    for fam in FAMILIES:
        if (a,b,fam) in done:
            continue
        seed = stable_seed("v2","B2000",a,b,fam)
        tc = time.time()
        r = run_cell(u,v,fam,B_MAIN,seed)
        rows.append({"pair":plabel(a,b),"a":a,"b":b,**r,"seed":seed})
        df = pd.DataFrame(rows).sort_values(["a","b","family"]).reset_index(drop=True)
        atomic_csv(df,CHECKPOINT)
        print(f"[{len(df):02d}/80] {plabel(a,b):18s} {fam:18s} "
              f"p={r['p_value']:.6f} ({time.time()-tc:.1f}s)")

main = pd.read_csv(CHECKPOINT)
if len(main) != 80:
    raise RuntimeError(f"Incomplete checkpoint: {len(main)}/80")

border = main[(main.p_value >= BORDER_LO) & (main.p_value <= BORDER_HI)]
print(f"\nBorderline cells: {len(border)}")
brows = []
for _, rr in border.iterrows():
    a,b,fam = rr["a"],rr["b"],rr["family"]
    u,v = cache[(a,b)]
    seed = stable_seed("v2","B5000",a,b,fam)
    tc=time.time()
    r=run_cell(u,v,fam,B_BORDER,seed)
    brows.append({"pair":plabel(a,b),"a":a,"b":b,**r,"seed":seed})
    print(f"[B5000] {plabel(a,b):18s} {fam:18s} "
          f"{rr.p_value:.6f} -> {r['p_value']:.6f} ({time.time()-tc:.1f}s)")

bdf=pd.DataFrame(brows)
bdf.to_csv(BORDERFILE,index=False,float_format="%.12g")

final=main.copy()
final["p_B2000"]=final["p_value"]
final["p_B5000"]=np.nan
final["final_p"]=final["p_B2000"]
final["final_B"]=B_MAIN
if len(bdf):
    lookup={(r.a,r.b,r.family):r for _,r in bdf.iterrows()}
    for i,r in final.iterrows():
        k=(r.a,r.b,r.family)
        if k in lookup:
            final.loc[i,"p_B5000"]=lookup[k].p_value
            final.loc[i,"final_p"]=lookup[k].p_value
            final.loc[i,"final_B"]=B_BORDER
final["reject_5pct"]=final.final_p < ALPHA
final["nonreject_5pct"]=~final.reject_5pct
final.to_csv(FINAL,index=False,float_format="%.12g")

psum=(final.groupby(["pair","a","b"],as_index=False)
      .agg(n_families=("family","size"),
           n_reject_5pct=("reject_5pct","sum"),
           n_nonreject_5pct=("nonreject_5pct","sum"),
           min_final_p=("final_p","min"),
           max_final_p=("final_p","max")))
psum.to_csv(PAIRSUM,index=False,float_format="%.12g")

commodity={"cocoa","gold","brent","wti"}
cc=final.a.isin(commodity)&final.b.isin(commodity)
cx=(final.a=="cedi")|(final.b=="cedi")
summary={
    "pit_sha256":sha256(PITFILE),
    "n_obs":len(pit),
    "n_cells":len(final),
    "borderline_initial_cells":len(border),
    "all_rejected":int(final.reject_5pct.sum()),
    "all_nonrejected":int(final.nonreject_5pct.sum()),
    "commodity_commodity_cells":int(cc.sum()),
    "commodity_commodity_rejected":int(final.loc[cc,"reject_5pct"].sum()),
    "commodity_cedi_cells":int(cx.sum()),
    "commodity_cedi_rejected":int(final.loc[cx,"reject_5pct"].sum()),
    "runtime_seconds":time.time()-t0,
}
SUMMARY.write_text(json.dumps(summary,indent=2),encoding="utf-8")

manifest=[PITFILE,FITS,EMP,CHECKPOINT,BORDERFILE,FINAL,PAIRSUM,SUMMARY]
with HASHES.open("w",encoding="utf-8",newline="\n") as f:
    for p in manifest:
        f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")

print("\n=== FINAL GOF PAIR SUMMARY ===")
print(psum.to_string(index=False))
print("\n=== STAGE 2 SUMMARY ===")
print(json.dumps(summary,indent=2))
print("\n=== OUTPUT HASHES ===")
print(HASHES.read_text(encoding="utf-8"))
print("=== PUBLICATION REBUILD V2 STAGE 2 COMPLETE ===")
