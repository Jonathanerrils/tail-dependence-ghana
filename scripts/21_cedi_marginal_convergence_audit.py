"""Audit convergence of every CURRENT Cedi marginal candidate.

This is a non-destructive audit. It fits each specification in the current
marginals.SPEC_LADDER directly with arch_model and reports:
- convergence_flag
- optimizer message
- warnings
- AIC/BIC
- whether the fit is finite

It writes a new audit CSV only.

Run from repository root:
    python scripts/21_cedi_marginal_convergence_audit.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk import marginals  # noqa: E402

RETURNS = ROOT / "data" / "processed" / "returns_real.csv"
OUT = ROOT / "outputs" / "tables" / "cedi_marginal_convergence_audit.csv"

rets = pd.read_csv(RETURNS, index_col=0, parse_dates=True)
r = rets["cedi"].dropna()

print("=== CEDI MARGINAL CONVERGENCE AUDIT ===")
print(f"n = {len(r)}")
print(f"date range = {r.index.min().date()} to {r.index.max().date()}")
print()

rows = []

for spec_name, spec in marginals.SPEC_LADDER:
    print(f"--- {spec_name} ---")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = arch_model(r, **spec).fit(disp="off")
            flag = int(res.convergence_flag)
            message = str(getattr(res.optimization_result, "message", ""))
            params_finite = bool(np.isfinite(np.asarray(res.params, dtype=float)).all())
            objective_finite = bool(np.isfinite(float(res.loglikelihood)))
            aic = float(res.aic)
            bic = float(res.bic)
            warning_text = " | ".join(str(w.message) for w in caught)

            print(f"convergence_flag : {flag}")
            print(f"message          : {message}")
            print(f"AIC              : {aic:.6f}")
            print(f"BIC              : {bic:.6f}")
            print(f"finite params    : {params_finite}")
            print(f"finite loglik    : {objective_finite}")
            print(f"warnings         : {warning_text if warning_text else 'None'}")

            valid = (flag == 0) and params_finite and objective_finite
            print(f"VALID FIT        : {valid}")

            rows.append({
                "spec": spec_name,
                "convergence_flag": flag,
                "optimizer_message": message,
                "aic": aic,
                "bic": bic,
                "params_finite": params_finite,
                "loglik_finite": objective_finite,
                "warnings": warning_text,
                "valid_fit": valid,
            })

        except Exception as exc:
            print(f"FIT ERROR        : {type(exc).__name__}: {exc}")
            rows.append({
                "spec": spec_name,
                "convergence_flag": np.nan,
                "optimizer_message": "",
                "aic": np.nan,
                "bic": np.nan,
                "params_finite": False,
                "loglik_finite": False,
                "warnings": " | ".join(str(w.message) for w in caught),
                "valid_fit": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    print()

df = pd.DataFrame(rows)
df.to_csv(
    OUT,
    index=False,
    float_format="%.12f",
    lineterminator="\n",
    encoding="utf-8",
)

print("=== SUMMARY ===")
print(df[["spec", "convergence_flag", "aic", "valid_fit"]].to_string(index=False))

invalid = df[~df["valid_fit"].fillna(False)]

print()
if len(invalid) == 0:
    print("All Cedi candidate fits converged with finite parameters/log-likelihood.")
else:
    print(f"INVALID/NON-CONVERGED CANDIDATES: {len(invalid)}")
    print(invalid[["spec", "convergence_flag", "optimizer_message"]].to_string(index=False))

print()
print(f"Wrote: {OUT.relative_to(ROOT)}")
print("=== CEDI CONVERGENCE AUDIT COMPLETE ===")
