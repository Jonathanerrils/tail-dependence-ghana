"""Publication marginal audit.

Audits the CURRENT local marginal implementation against current canonical
daily returns and the frozen residual/PIT files. It does not overwrite any
existing publication artifact.

Run from repository root:
    python scripts/20_publication_marginal_audit.py
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk import marginals, inference  # noqa: E402

TAB = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"

RETURNS = PROC / "returns_real.csv"
FROZEN_RESID = TAB / "_resid_real.csv"
FROZEN_PIT = TAB / "_pit_real.csv"

OUT_SUMMARY = TAB / "marginal_audit_current.csv"
OUT_SEARCH = TAB / "marginal_audit_spec_search.csv"
OUT_COMPARE = TAB / "marginal_audit_frozen_compare.csv"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not RETURNS.exists():
    raise FileNotFoundError(f"Missing canonical returns: {RETURNS}")

rets = pd.read_csv(RETURNS, index_col=0, parse_dates=True)

print("=== INPUT FREEZE ===")
print(f"returns_real.csv SHA256: {sha256(RETURNS)}")
print(f"marginals.py SHA256:     {sha256(ROOT / 'src' / 'tailrisk' / 'marginals.py')}")
print(f"inference.py SHA256:     {sha256(ROOT / 'src' / 'tailrisk' / 'inference.py')}")
if FROZEN_RESID.exists():
    print(f"_resid_real.csv SHA256:  {sha256(FROZEN_RESID)}")
if FROZEN_PIT.exists():
    print(f"_pit_real.csv SHA256:    {sha256(FROZEN_PIT)}")
print(f"Returns shape:           {rets.shape}")
print(f"Date range:              {rets.index.min().date()} to {rets.index.max().date()}")
print(f"Columns:                 {list(rets.columns)}")

print("\n=== REFITTING CURRENT DAILY MARGINAL LADDER (AUDIT ONLY) ===")
fits, search_log = marginals.fit_all(rets, gate=True)

rows = []
for name, fit in fits.items():
    diag = inference.marginal_diagnostics(fit.std_resid, fit.pit)
    rows.append({
        "series": name,
        "spec": getattr(fit, "spec", ""),
        "adequate": bool(getattr(fit, "adequate", diag.get("adequate_5pct", False))),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "nu": float(fit.nu) if np.isfinite(fit.nu) else np.nan,
        **diag,
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_SUMMARY, index=False, float_format="%.12f",
               lineterminator="\n", encoding="utf-8")

search = pd.DataFrame(search_log)
search.to_csv(OUT_SEARCH, index=False, float_format="%.12f",
              lineterminator="\n", encoding="utf-8")

print("\n=== CURRENT DAILY MARGINAL ADEQUACY ===")
cols = [c for c in ["series","spec","adequate","aic","lb_resid_p",
                    "lb_sq_p","ks_pit_p","adequate_5pct"] if c in summary.columns]
print(summary[cols].to_string(index=False))

fresh_resid = marginals.std_resid_frame(fits)
fresh_pit = marginals.pit_frame(fits)
compare_rows = []

def compare_frame(kind, fresh, path):
    if not path.exists():
        print(f"\n{kind}: frozen file not found; skipped.")
        return
    old = pd.read_csv(path, index_col=0, parse_dates=True)
    common_cols = [c for c in fresh.columns if c in old.columns]
    common_idx = fresh.index.intersection(old.index)
    print(f"\n=== {kind.upper()} FRESH VS FROZEN ===")
    print(f"Fresh shape: {fresh.shape}; frozen shape: {old.shape}")
    print(f"Common dates: {len(common_idx)}; common columns: {common_cols}")
    for c in common_cols:
        a = fresh.loc[common_idx, c].astype(float)
        b = old.loc[common_idx, c].astype(float)
        mask = a.notna() & b.notna()
        aa, bb = a[mask], b[mask]
        if len(aa):
            corr = float(aa.corr(bb))
            diff = (aa - bb).abs()
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            exact = bool(np.array_equal(aa.to_numpy(), bb.to_numpy()))
        else:
            corr = max_abs = mean_abs = np.nan
            exact = False
        compare_rows.append({
            "kind": kind, "series": c, "n_common": int(len(aa)),
            "correlation": corr, "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs, "exact_values": exact,
        })
        print(f"{c:8s} n={len(aa):4d} corr={corr:.12f} "
              f"max|diff|={max_abs:.12g} mean|diff|={mean_abs:.12g} exact={exact}")

compare_frame("residual", fresh_resid, FROZEN_RESID)
compare_frame("pit", fresh_pit, FROZEN_PIT)

pd.DataFrame(compare_rows).to_csv(
    OUT_COMPARE, index=False, float_format="%.12f",
    lineterminator="\n", encoding="utf-8"
)

print("\n=== AUDIT OUTPUTS ===")
print(OUT_SUMMARY.relative_to(ROOT))
print(OUT_SEARCH.relative_to(ROOT))
print(OUT_COMPARE.relative_to(ROOT))

print("\n=== INTERPRETATION RULE ===")
print("Passing adequacy means all pre-declared diagnostics pass at 5%.")
print("If no candidate passes, that series remains UNRESOLVED.")
print("AIC ranking among inadequate candidates does not convert failure into adequacy.")
print("Fresh-vs-frozen disagreement means downstream artifacts were produced")
print("under a different code/model state and must be traced before manuscript use.")

print("\n=== MARGINAL AUDIT COMPLETE ===")
print("No existing publication output was overwritten.")
