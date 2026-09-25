"""Publication rebuild v2 — Stage 1: synchronized panel, marginals, EVT.

NON-DESTRUCTIVE:
- Reads the synchronized common-interval candidate produced by script 27.
- Does NOT overwrite data/processed/returns_real.csv or legacy publication outputs.
- Writes all new artifacts under outputs/publication_rebuild_v2/tables/.

Scientific decisions implemented here:
1. Candidate marginal ladder is fixed:
   AR1-GJR-GARCH-t
   AR2-GJR-GARCH-t
   AR1-EGARCH-t
   AR1-GJR-GARCH-skewt
2. A candidate is VALID only if:
   - optimizer convergence_flag == 0,
   - loglikelihood and all fitted parameters are finite,
   - standardized residuals, conditional volatility and PIT values are finite.
3. The first VALID candidate passing all three diagnostics at 5% is selected.
4. If no VALID candidate passes, the lowest-AIC VALID candidate is retained only
   as an INADEQUATE deterministic reference.
5. Non-converged candidates never enter the adequacy/AIC fallback pool.
6. PIT values are computed with the fitted arch distribution object's own CDF
   and its declared distribution parameter names.
7. EVT is applied to the negative standardized residuals of the selected fits.

Run from repository root:
    python scripts/28_publication_rebuild_stage1.py
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
sys.path.insert(0, str(SRC))

from tailrisk import evt, inference  # noqa: E402

INFILE = ROOT / "outputs" / "tables" / "returns_real_synchronized_common_candidate.csv"
OUT = ROOT / "outputs" / "publication_rebuild_v2" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
LADDER = [
    ("AR1-GJR-GARCH-t",
     dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR2-GJR-GARCH-t",
     dict(mean="AR", lags=2, vol="GARCH", p=1, o=1, q=1, dist="t")),
    ("AR1-EGARCH-t",
     dict(mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t")),
    ("AR1-GJR-GARCH-skewt",
     dict(mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt")),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fit_candidate(x: pd.Series, series: str, spec_name: str, spec: dict):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = arch_model(x.dropna(), **spec).fit(disp="off")

    conv = int(getattr(res, "convergence_flag", -999))
    opt = getattr(res, "optimization_result", None)
    opt_msg = str(getattr(opt, "message", "")) if opt is not None else ""

    std_resid = (res.resid / res.conditional_volatility).dropna()
    cond_vol = res.conditional_volatility.reindex(std_resid.index)

    dist = res.model.distribution
    pnames = list(dist.parameter_names())
    dparams = [float(res.params[p]) for p in pnames]
    pit_vals = np.asarray(dist.cdf(std_resid.to_numpy(), dparams), dtype=float)
    pit = pd.Series(
        np.clip(pit_vals, 1e-6, 1 - 1e-6),
        index=std_resid.index,
        name=series,
    )

    finite = (
        np.isfinite(float(res.loglikelihood))
        and np.isfinite(np.asarray(res.params, dtype=float)).all()
        and np.isfinite(std_resid.to_numpy()).all()
        and np.isfinite(cond_vol.to_numpy()).all()
        and np.isfinite(pit.to_numpy()).all()
    )
    valid = (conv == 0) and finite

    diag = inference.marginal_diagnostics(std_resid, pit) if valid else {
        "lb_resid_p": np.nan,
        "lb_sq_p": np.nan,
        "ks_pit_p": np.nan,
        "adequate_5pct": False,
    }

    return {
        "series": series,
        "spec": spec_name,
        "result": res,
        "std_resid": std_resid.rename(series),
        "cond_vol": cond_vol.rename(series),
        "pit": pit,
        "convergence_flag": conv,
        "optimizer_message": opt_msg,
        "finite_fit": bool(finite),
        "valid_fit": bool(valid),
        "warnings": " | ".join(str(w.message) for w in caught),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglikelihood": float(res.loglikelihood),
        "dist_param_names": ",".join(pnames),
        "dist_param_values": ",".join(f"{v:.12g}" for v in dparams),
        **diag,
    }

if not INFILE.exists():
    raise FileNotFoundError(
        f"{INFILE} not found. Run scripts/27_synchronized_common_interval_audit.py first."
    )

returns = pd.read_csv(INFILE, index_col=0, parse_dates=True).sort_index()
if list(returns.columns) != ["cocoa", "gold", "brent", "wti", "cedi"]:
    raise ValueError(f"Unexpected columns: {list(returns.columns)}")

print("=== PUBLICATION REBUILD V2: STAGE 1 ===")
print(f"input: {INFILE.relative_to(ROOT)}")
print(f"input SHA256: {sha256(INFILE)}")
print(f"shape: {returns.shape}")
print(f"period: {returns.index.min().date()} to {returns.index.max().date()}")
print()

candidate_rows = []
selected = {}

for series in returns.columns:
    print(f"--- {series.upper()} ---")
    candidates = []

    for spec_name, spec in LADDER:
        try:
            c = fit_candidate(returns[series], series, spec_name, spec)
            candidates.append(c)

            print(
                f"{spec_name:23s} "
                f"conv={c['convergence_flag']} "
                f"valid={c['valid_fit']} "
                f"AIC={c['aic']:.6f} "
                f"LBr={c['lb_resid_p'] if np.isfinite(c['lb_resid_p']) else np.nan:.6g} "
                f"LBsq={c['lb_sq_p'] if np.isfinite(c['lb_sq_p']) else np.nan:.6g} "
                f"KS={c['ks_pit_p'] if np.isfinite(c['ks_pit_p']) else np.nan:.6g} "
                f"adequate={bool(c['adequate_5pct'])}"
            )
        except Exception as e:
            c = {
                "series": series,
                "spec": spec_name,
                "convergence_flag": np.nan,
                "optimizer_message": "",
                "finite_fit": False,
                "valid_fit": False,
                "warnings": "",
                "aic": np.nan,
                "bic": np.nan,
                "loglikelihood": np.nan,
                "dist_param_names": "",
                "dist_param_values": "",
                "lb_resid_p": np.nan,
                "lb_sq_p": np.nan,
                "ks_pit_p": np.nan,
                "adequate_5pct": False,
                "error": repr(e),
            }
            candidates.append(c)
            print(f"{spec_name:23s} ERROR: {e!r}")

        candidate_rows.append({
            k: v for k, v in c.items()
            if k not in {"result", "std_resid", "cond_vol", "pit"}
        })

    adequate = [
        c for c in candidates
        if c.get("valid_fit", False) and bool(c.get("adequate_5pct", False))
        and "result" in c
    ]

    if adequate:
        chosen = adequate[0]
        status = "ADEQUATE"
        selection_rule = "first valid adequate candidate in fixed ladder"
    else:
        valid = [
            c for c in candidates
            if c.get("valid_fit", False) and np.isfinite(c.get("aic", np.nan))
            and "result" in c
        ]
        if not valid:
            raise RuntimeError(f"No valid converged candidate for {series}")
        chosen = min(valid, key=lambda z: z["aic"])
        status = "INADEQUATE_REFERENCE"
        selection_rule = "lowest AIC among valid converged candidates; no candidate passed gate"

    chosen["selection_status"] = status
    chosen["selection_rule"] = selection_rule
    selected[series] = chosen
    print(f"SELECTED: {chosen['spec']} [{status}]")
    print()

candidate_df = pd.DataFrame(candidate_rows)
candidate_df.to_csv(OUT / "marginal_candidate_audit.csv", index=False, float_format="%.12g")

selected_rows = []
for series, c in selected.items():
    selected_rows.append({
        "series": series,
        "spec": c["spec"],
        "selection_status": c["selection_status"],
        "selection_rule": c["selection_rule"],
        "convergence_flag": c["convergence_flag"],
        "valid_fit": c["valid_fit"],
        "aic": c["aic"],
        "bic": c["bic"],
        "loglikelihood": c["loglikelihood"],
        "lb_resid_p": c["lb_resid_p"],
        "lb_sq_p": c["lb_sq_p"],
        "ks_pit_p": c["ks_pit_p"],
        "adequate_5pct": c["adequate_5pct"],
        "dist_param_names": c["dist_param_names"],
        "dist_param_values": c["dist_param_values"],
    })
selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv(OUT / "marginal_selected.csv", index=False, float_format="%.12g")

resid = pd.concat([selected[c]["std_resid"] for c in returns.columns], axis=1).dropna()
pit = pd.concat([selected[c]["pit"] for c in returns.columns], axis=1).dropna()

if not resid.index.equals(pit.index):
    common = resid.index.intersection(pit.index)
    resid = resid.loc[common]
    pit = pit.loc[common]

resid.to_csv(OUT / "_resid_publication_v2.csv")
pit.to_csv(OUT / "_pit_publication_v2.csv")

print("=== ALIGNED SELECTED RESIDUAL/PIT PANEL ===")
print(f"residuals: {resid.shape}, {resid.index.min().date()} to {resid.index.max().date()}")
print(f"PITs:      {pit.shape}, {pit.index.min().date()} to {pit.index.max().date()}")
print()

# EVT q90 for all selected residual series
evt_rows = []
for series in resid.columns:
    losses = -resid[series].dropna()
    pot = evt.fit_pot(losses, quantile=0.90)
    k = max(25, int(0.05 * losses.size))
    evt_rows.append({
        "series": series,
        "n_total": int(losses.size),
        "threshold_q": 0.90,
        "threshold": float(pot.threshold),
        "xi": float(pot.xi),
        "beta": float(pot.beta),
        "n_exceed": int(pot.n_exceed),
        "VaR99_resid": float(pot.var(0.99)),
        "ES99_resid": float(pot.es(0.99)),
        "hill_gamma_k5pct": float(evt.hill_estimator(losses, k)),
        "hill_k": int(k),
    })

evt_df = pd.DataFrame(evt_rows)
evt_df.to_csv(OUT / "evt_pot_gpd_publication_v2.csv", index=False, float_format="%.12g")

# Cedi threshold sensitivity
cedi_losses = -resid["cedi"].dropna()
sens_rows = []
for q in [0.85, 0.875, 0.90, 0.925, 0.95, 0.96, 0.97, 0.975]:
    pot = evt.fit_pot(cedi_losses, quantile=q)
    sens_rows.append({
        "quantile": q,
        "n_total": int(cedi_losses.size),
        "threshold": float(pot.threshold),
        "xi": float(pot.xi),
        "beta": float(pot.beta),
        "n_exceed": int(pot.n_exceed),
        "VaR99_resid": float(pot.var(0.99)),
        "ES99_resid": float(pot.es(0.99)),
    })
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(
    OUT / "cedi_evt_threshold_sensitivity_publication_v2.csv",
    index=False, float_format="%.12g"
)

# Summary
summary = {
    "input_file": str(INFILE.relative_to(ROOT)),
    "input_sha256": sha256(INFILE),
    "returns_shape": list(returns.shape),
    "returns_period": [str(returns.index.min().date()), str(returns.index.max().date())],
    "resid_shape": list(resid.shape),
    "pit_shape": list(pit.shape),
    "selected_marginals": {
        s: {
            "spec": selected[s]["spec"],
            "status": selected[s]["selection_status"],
            "convergence_flag": selected[s]["convergence_flag"],
            "lb_resid_p": selected[s]["lb_resid_p"],
            "lb_sq_p": selected[s]["lb_sq_p"],
            "ks_pit_p": selected[s]["ks_pit_p"],
        }
        for s in returns.columns
    },
}
(OUT / "stage1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Hash manifest after all outputs are written.
manifest_files = [
    INFILE,
    OUT / "marginal_candidate_audit.csv",
    OUT / "marginal_selected.csv",
    OUT / "_resid_publication_v2.csv",
    OUT / "_pit_publication_v2.csv",
    OUT / "evt_pot_gpd_publication_v2.csv",
    OUT / "cedi_evt_threshold_sensitivity_publication_v2.csv",
    OUT / "stage1_summary.json",
]
with (OUT / "stage1_hashes.txt").open("w", encoding="utf-8", newline="\n") as f:
    for p in manifest_files:
        f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")

print("=== SELECTED MARGINALS ===")
print(selected_df.to_string(index=False))
print()
print("=== EVT Q90 ===")
print(evt_df.to_string(index=False))
print()
print("=== CEDI EVT THRESHOLD SENSITIVITY ===")
print(sens_df.to_string(index=False))
print()
print("=== OUTPUT HASHES ===")
print((OUT / "stage1_hashes.txt").read_text(encoding="utf-8"))
print("=== PUBLICATION REBUILD V2 STAGE 1 COMPLETE ===")
