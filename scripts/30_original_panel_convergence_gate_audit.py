"""Audit all marginal candidates on the frozen original 3,008-row panel.

Non-destructive. It does not overwrite canonical data or any existing
publication results.

Run:
    python scripts/30_original_panel_convergence_gate_audit.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tailrisk import inference  # noqa: E402

INFILE = ROOT / "data" / "processed" / "returns_real.csv"
OUT = ROOT / "outputs" / "original_panel_gate_audit"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA = "3e9769f42ec7925aacd951277ddae6ff529822408815b01210de87659221341f"
EXPECTED_SHAPE = (3008, 5)
EXPECTED_COLUMNS = ["cocoa", "gold", "brent", "wti", "cedi"]
EXPECTED_START = "2015-01-05"
EXPECTED_END = "2026-07-15"

LADDER = [
    ("AR1-GJR-GARCH-t", dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR2-GJR-GARCH-t", dict(mean="AR", lags=2, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR1-EGARCH-t", dict(mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t")),
    ("AR1-GJR-GARCH-skewt", dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt")),
]

PREVIOUS = {
    "cocoa": ("AR1-GJR-GARCH-t", "ADEQUATE"),
    "gold": ("AR1-GJR-GARCH-t", "ADEQUATE"),
    "brent": ("AR1-GJR-GARCH-skewt", "ADEQUATE"),
    "wti": ("AR1-EGARCH-t", "ADEQUATE"),
    "cedi": ("AR1-GJR-GARCH-t", "INADEQUATE_REFERENCE"),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fit_one(x, series, spec_name, spec):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = arch_model(x.dropna(), **spec).fit(disp="off")

    conv = int(getattr(res, "convergence_flag", -999))
    opt = getattr(res, "optimization_result", None)
    opt_success = bool(getattr(opt, "success", conv == 0))
    opt_message = str(getattr(opt, "message", ""))

    z = (res.resid / res.conditional_volatility).dropna()
    vol = res.conditional_volatility.reindex(z.index)

    dist = res.model.distribution
    pnames = list(dist.parameter_names())
    dparams = [float(res.params[p]) for p in pnames]
    pit = pd.Series(
        np.clip(np.asarray(dist.cdf(z.to_numpy(), dparams), dtype=float), 1e-6, 1 - 1e-6),
        index=z.index,
        name=series,
    )

    finite = bool(
        np.isfinite(float(res.loglikelihood))
        and np.isfinite(np.asarray(res.params, dtype=float)).all()
        and np.isfinite(z.to_numpy()).all()
        and np.isfinite(vol.to_numpy()).all()
        and (vol.to_numpy() > 0).all()
        and np.isfinite(pit.to_numpy()).all()
    )
    valid = bool(conv == 0 and finite)

    diag = (
        inference.marginal_diagnostics(z, pit)
        if valid else
        {"lb_resid_p": np.nan, "lb_sq_p": np.nan, "ks_pit_p": np.nan, "adequate_5pct": False}
    )

    return {
        "series": series,
        "spec": spec_name,
        "convergence_flag": conv,
        "optimizer_success": opt_success,
        "optimizer_message": opt_message,
        "finite_fit": finite,
        "valid_fit": valid,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglikelihood": float(res.loglikelihood),
        "lb_resid_p": diag["lb_resid_p"],
        "lb_sq_p": diag["lb_sq_p"],
        "ks_pit_p": diag["ks_pit_p"],
        "adequate_5pct": bool(diag["adequate_5pct"]),
        "dist_param_names": ",".join(pnames),
        "dist_param_values": ",".join(f"{v:.12g}" for v in dparams),
        "warnings": " | ".join(str(w.message) for w in caught),
    }

if not INFILE.exists():
    raise FileNotFoundError(INFILE)

actual_sha = sha256(INFILE)
returns = pd.read_csv(INFILE, index_col=0, parse_dates=True).sort_index()

print("=== ORIGINAL PANEL HARD CONVERGENCE-GATE AUDIT ===")
print(f"input: {INFILE.relative_to(ROOT)}")
print(f"SHA256: {actual_sha}")
print(f"shape: {returns.shape}")
print(f"period: {returns.index.min().date()} to {returns.index.max().date()}")
print()

checks = [
    actual_sha == EXPECTED_SHA,
    tuple(returns.shape) == EXPECTED_SHAPE,
    list(returns.columns) == EXPECTED_COLUMNS,
    str(returns.index.min().date()) == EXPECTED_START,
    str(returns.index.max().date()) == EXPECTED_END,
]
if not all(checks):
    raise SystemExit("Freeze check failed. This is not the frozen original 3,008-row panel.")

print("Freeze check: PASS")
print()

candidate_rows = []
selected_rows = []
compare_rows = []

for series in returns.columns:
    print(f"--- {series.upper()} ---")
    cand = []
    for spec_name, spec in LADDER:
        try:
            r = fit_one(returns[series], series, spec_name, spec)
        except Exception as e:
            r = {
                "series": series, "spec": spec_name,
                "convergence_flag": np.nan, "optimizer_success": False,
                "optimizer_message": "", "finite_fit": False, "valid_fit": False,
                "aic": np.nan, "bic": np.nan, "loglikelihood": np.nan,
                "lb_resid_p": np.nan, "lb_sq_p": np.nan, "ks_pit_p": np.nan,
                "adequate_5pct": False, "dist_param_names": "",
                "dist_param_values": "", "warnings": "", "error": repr(e),
            }

        cand.append(r)
        candidate_rows.append(r)

        def ff(x):
            return "nan" if not np.isfinite(x) else f"{x:.6g}"

        print(
            f"{spec_name:23s} conv={r['convergence_flag']} "
            f"valid={r['valid_fit']} AIC={ff(r['aic'])} "
            f"LBr={ff(r['lb_resid_p'])} LBsq={ff(r['lb_sq_p'])} "
            f"KS={ff(r['ks_pit_p'])} adequate={r['adequate_5pct']}"
        )

    adequate = [r for r in cand if r["valid_fit"] and r["adequate_5pct"]]
    if adequate:
        chosen = adequate[0]
        status = "ADEQUATE"
        rule = "first valid adequate candidate in fixed ladder"
    else:
        valid = [r for r in cand if r["valid_fit"] and np.isfinite(r["aic"])]
        if not valid:
            raise RuntimeError(f"No valid converged candidate for {series}")
        chosen = min(valid, key=lambda r: r["aic"])
        status = "INADEQUATE_REFERENCE"
        rule = "lowest AIC among valid converged candidates; no valid candidate passed adequacy"

    old_spec, old_status = PREVIOUS[series]
    old_row = next(r for r in cand if r["spec"] == old_spec)
    changed = chosen["spec"] != old_spec or status != old_status

    selected_rows.append({
        "series": series,
        "selected_spec_hard_gate": chosen["spec"],
        "selection_status_hard_gate": status,
        "selection_rule": rule,
        "convergence_flag": chosen["convergence_flag"],
        "valid_fit": chosen["valid_fit"],
        "aic": chosen["aic"],
        "lb_resid_p": chosen["lb_resid_p"],
        "lb_sq_p": chosen["lb_sq_p"],
        "ks_pit_p": chosen["ks_pit_p"],
        "adequate_5pct": chosen["adequate_5pct"],
    })
    compare_rows.append({
        "series": series,
        "previous_spec": old_spec,
        "previous_status": old_status,
        "previous_spec_convergence_flag_now": old_row["convergence_flag"],
        "previous_spec_valid_now": old_row["valid_fit"],
        "previous_spec_adequate_now": old_row["adequate_5pct"],
        "hard_gate_spec": chosen["spec"],
        "hard_gate_status": status,
        "selection_changed": changed,
    })

    print(f"PREVIOUS: {old_spec} [{old_status}] conv={old_row['convergence_flag']} valid={old_row['valid_fit']}")
    print(f"HARD GATE: {chosen['spec']} [{status}] changed={changed}")
    print()

cand_df = pd.DataFrame(candidate_rows)
sel_df = pd.DataFrame(selected_rows)
cmp_df = pd.DataFrame(compare_rows)

cand_path = OUT / "original_panel_marginal_candidate_audit.csv"
sel_path = OUT / "original_panel_marginal_selected_hard_gate.csv"
cmp_path = OUT / "original_panel_selection_comparison.csv"
summary_path = OUT / "original_panel_gate_audit_summary.json"
hash_path = OUT / "original_panel_gate_audit_hashes.txt"

cand_df.to_csv(cand_path, index=False, float_format="%.12g")
sel_df.to_csv(sel_path, index=False, float_format="%.12g")
cmp_df.to_csv(cmp_path, index=False)

invalid = cand_df[~cand_df["valid_fit"].astype(bool)]
changed = cmp_df[cmp_df["selection_changed"].astype(bool)]

summary = {
    "input_sha256": actual_sha,
    "freeze_check_passed": True,
    "n_candidate_fits": int(len(cand_df)),
    "n_invalid_or_nonconverged_candidates": int(len(invalid)),
    "invalid_candidates": [
        {
            "series": r["series"],
            "spec": r["spec"],
            "convergence_flag": None if pd.isna(r["convergence_flag"]) else int(r["convergence_flag"]),
            "optimizer_message": str(r.get("optimizer_message", "")),
        }
        for _, r in invalid.iterrows()
    ],
    "n_selections_changed_vs_previously_reported": int(len(changed)),
    "changed_series": changed["series"].tolist(),
    "all_previously_reported_selected_models_valid_now": bool(cmp_df["previous_spec_valid_now"].all()),
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

with hash_path.open("w", encoding="utf-8", newline="\n") as f:
    for p in [INFILE, ROOT/"src"/"tailrisk"/"inference.py", cand_path, sel_path, cmp_path, summary_path]:
        f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")

print("=== HARD-GATE SELECTION COMPARISON ===")
print(cmp_df.to_string(index=False))
print()

print("=== INVALID / NON-CONVERGED CANDIDATES ===")
if len(invalid):
    print(invalid[["series","spec","convergence_flag","optimizer_success","optimizer_message","finite_fit","valid_fit"]].to_string(index=False))
else:
    print("none")
print()

print("=== AUDIT SUMMARY ===")
print(json.dumps(summary, indent=2))
print()

print("=== OUTPUT HASHES ===")
print(hash_path.read_text(encoding="utf-8"))
print("=== ORIGINAL PANEL HARD CONVERGENCE-GATE AUDIT COMPLETE ===")
