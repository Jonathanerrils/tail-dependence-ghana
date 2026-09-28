"""SAFE Panel A Cedi de-clipped rank / observed-GOF sensitivity audit v2.

Uses the actual local Panel A residual file:
    outputs/tables/_resid_real.csv

Purpose
-------
The Panel A Cedi PIT has three values clipped to 1e-6. Since the copula
pipeline subsequently rank-transforms PITs, clipping replaces the true ordering
of those three observations with a three-way average rank.

For a continuous monotone marginal CDF, the ordering of the unclipped PITs is
the same as the ordering of the standardized residuals. This script therefore
uses the saved standardized residuals to recover the actual rank order and
measures the effect on the 32 Panel A commodity-Cedi observed GOF cells.

Safety
------
- NO bootstrap
- NO marginal model fitting
- NO recursive output scanning
- reads only two CSVs
- writes only to outputs/pit_ties_audit/
- does not overwrite publication outputs

Run from repository root:
    python scripts/36_panelA_cedi_declipped_rank_gof_sensitivity_SAFE_v2.py
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk import copulas, gof  # noqa: E402

PIT = ROOT / "outputs" / "tables" / "_pit_real.csv"
RESID = ROOT / "outputs" / "tables" / "_resid_real.csv"
OUT = ROOT / "outputs" / "pit_ties_audit"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_PIT_SHA = "be42455c12bf2f4a9c7ef33494930ad396f569192f571386d099046a388119d1"
EXPECTED_SHAPE = (3007, 5)
EXPECTED_START = "2015-01-06"
EXPECTED_END = "2026-07-15"

FLOOR = 1e-6
FAMILIES = [
    "gaussian", "t", "clayton", "gumbel",
    "frank", "joe", "survival_clayton", "survival_gumbel",
]
COMMODITIES = ["cocoa", "gold", "brent", "wti"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def max_param_diff(p1: dict, p2: dict) -> float:
    diffs = []
    for key in sorted(set(p1) | set(p2)):
        if key not in p1 or key not in p2:
            return float("inf")
        a, b = float(p1[key]), float(p2[key])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(abs(a - b))
        elif a != b:
            return float("inf")
    return max(diffs) if diffs else 0.0


for path in (PIT, RESID):
    if not path.exists():
        raise FileNotFoundError(path)

pit = pd.read_csv(PIT, index_col=0, parse_dates=True).sort_index()
resid = pd.read_csv(RESID, index_col=0, parse_dates=True).sort_index()

pit_sha = sha256(PIT)
resid_sha = sha256(RESID)

print("=== PANEL A CEDI DE-CLIPPED RANK / GOF SENSITIVITY AUDIT v2 ===")
print("No bootstrap will be run.")
print()
print(f"PIT path:       {PIT.relative_to(ROOT)}")
print(f"PIT SHA256:     {pit_sha}")
print(f"PIT freeze:     {pit_sha == EXPECTED_PIT_SHA}")
print(f"PIT shape:      {pit.shape}")
print(f"PIT period:     {pit.index.min().date()} to {pit.index.max().date()}")
print()
print(f"Residual path:  {RESID.relative_to(ROOT)}")
print(f"Residual SHA:   {resid_sha}")
print(f"Residual shape: {resid.shape}")
print(f"Residual period:{resid.index.min().date()} to {resid.index.max().date()}")
print()

# Hard geometry gate.
if pit.shape != EXPECTED_SHAPE:
    raise RuntimeError(f"Unexpected PIT shape {pit.shape}; expected {EXPECTED_SHAPE}.")
if str(pit.index.min().date()) != EXPECTED_START or str(pit.index.max().date()) != EXPECTED_END:
    raise RuntimeError("Unexpected PIT date range.")
if resid.shape != EXPECTED_SHAPE:
    raise RuntimeError(
        f"Unexpected residual shape {resid.shape}; expected {EXPECTED_SHAPE}."
    )
if not pit.index.equals(resid.index):
    raise RuntimeError("PIT and residual indexes are not identical.")

required = set(COMMODITIES + ["cedi"])
if not required.issubset(pit.columns):
    raise RuntimeError(f"PIT missing columns: {sorted(required - set(pit.columns))}")
if not required.issubset(resid.columns):
    raise RuntimeError(
        f"Residual file missing columns: {sorted(required - set(resid.columns))}"
    )

n = len(pit)
cedi_pit = pit["cedi"].to_numpy(float)
cedi_resid = resid["cedi"].to_numpy(float)

if not np.all(np.isfinite(cedi_resid)):
    raise RuntimeError("Cedi standardized residuals contain non-finite values.")

floor_idx = np.where(cedi_pit == FLOOR)[0]
if len(floor_idx) != 3:
    raise RuntimeError(f"Expected 3 Cedi PIT values at 1e-6; found {len(floor_idx)}.")

# Current rank input after clipping.
cedi_u_current = rankdata(cedi_pit, method="average") / (n + 1.0)

# Recovered rank input using the monotone standardized-residual ordering.
cedi_u_recovered = rankdata(cedi_resid, method="average") / (n + 1.0)

print("--- Three clipped observations ---")
floor_rows = []
for i in floor_idx:
    row = {
        "date": str(pit.index[i].date()),
        "pit": float(cedi_pit[i]),
        "std_resid": float(cedi_resid[i]),
        "current_rank": float(cedi_u_current[i] * (n + 1.0)),
        "recovered_rank": float(cedi_u_recovered[i] * (n + 1.0)),
        "current_u": float(cedi_u_current[i]),
        "recovered_u": float(cedi_u_recovered[i]),
    }
    floor_rows.append(row)
    print(
        f"{row['date']} | residual={row['std_resid']:.12g} | "
        f"current rank={row['current_rank']:.1f} | "
        f"recovered rank={row['recovered_rank']:.1f} | "
        f"recovered u={row['recovered_u']:.12g}"
    )
print()

# Sanity check: only the three clipped observations should change if the saved
# residuals and PITs originate from the same continuous marginal transform.
delta_u = np.abs(cedi_u_current - cedi_u_recovered)
changed_idx = np.where(delta_u > 0)[0]

print("--- Rank correction footprint ---")
print(f"observations with changed Cedi pseudo-u: {len(changed_idx)}")
print(f"maximum absolute pseudo-u change: {float(delta_u.max()):.12g}")
print(
    "all changed observations are the three clipped rows: "
    f"{set(changed_idx).issubset(set(floor_idx))}"
)
print()

if not set(changed_idx).issubset(set(floor_idx)):
    raise RuntimeError(
        "Recovered residual ranks differ from clipped-PIT ranks beyond the "
        "three floor observations. Stop and inspect provenance before GOF comparison."
    )

# Stress-tail membership.
print("--- Stress-threshold membership ---")
threshold_rows = []
for q in (0.025, 0.05, 0.10):
    current = cedi_u_current <= q
    recovered = cedi_u_recovered <= q
    changes = int(np.sum(current != recovered))
    threshold_rows.append({"q": q, "membership_changes": changes})
    print(f"q={q:.3f}: membership changes = {changes}")
print()

# Observed GOF sensitivity for only the 32 Cedi cells.
rows = []
print("--- Observed GOF sensitivity: 32 Panel A Cedi cells ---")
for commodity in COMMODITIES:
    commodity_u = rankdata(
        pit[commodity].to_numpy(float), method="average"
    ) / (n + 1.0)

    for family in FAMILIES:
        fit0 = copulas.FAMILIES[family](commodity_u, cedi_u_current)
        fit1 = copulas.FAMILIES[family](commodity_u, cedi_u_recovered)

        sn0 = float(
            gof.cvm_rosenblatt(
                commodity_u, cedi_u_current, family, fit0.params, grid_size=50
            )
        )
        sn1 = float(
            gof.cvm_rosenblatt(
                commodity_u, cedi_u_recovered, family, fit1.params, grid_size=50
            )
        )

        rows.append({
            "commodity": commodity,
            "family": family,
            "Sn_current": sn0,
            "Sn_recovered": sn1,
            "abs_Sn_change": abs(sn1 - sn0),
            "max_abs_parameter_change": max_param_diff(fit0.params, fit1.params),
            "current_params": json.dumps(fit0.params, sort_keys=True),
            "recovered_params": json.dumps(fit1.params, sort_keys=True),
        })

result = pd.DataFrame(rows)
detail_path = OUT / "panelA_cedi_declipped_observed_gof_sensitivity_v2.csv"
result.to_csv(detail_path, index=False, float_format="%.12g")

pair_summary = (
    result.groupby("commodity", as_index=False)
    .agg(
        max_abs_Sn_change=("abs_Sn_change", "max"),
        mean_abs_Sn_change=("abs_Sn_change", "mean"),
        max_abs_parameter_change=("max_abs_parameter_change", "max"),
    )
)

print(pair_summary.to_string(index=False))
print()

print("--- Cocoa-Cedi Frank ---")
cf = result[
    (result["commodity"] == "cocoa")
    & (result["family"] == "frank")
]
print(cf[
    [
        "commodity", "family", "Sn_current", "Sn_recovered",
        "abs_Sn_change", "max_abs_parameter_change"
    ]
].to_string(index=False))
print()

print("--- Ten largest observed Sn changes ---")
print(
    result.sort_values("abs_Sn_change", ascending=False).head(10)[
        [
            "commodity", "family", "Sn_current", "Sn_recovered",
            "abs_Sn_change", "max_abs_parameter_change"
        ]
    ].to_string(index=False)
)
print()

summary = {
    "pit_sha256": pit_sha,
    "pit_freeze_match": pit_sha == EXPECTED_PIT_SHA,
    "residual_sha256": resid_sha,
    "pit_shape": list(pit.shape),
    "residual_shape": list(resid.shape),
    "floor_observations": floor_rows,
    "changed_pseudo_u_observations": int(len(changed_idx)),
    "max_abs_pseudo_u_change": float(delta_u.max()),
    "tail_membership": threshold_rows,
    "max_abs_Sn_change_all_32": float(result["abs_Sn_change"].max()),
    "mean_abs_Sn_change_all_32": float(result["abs_Sn_change"].mean()),
    "max_abs_parameter_change_all_32": float(
        result["max_abs_parameter_change"].max()
    ),
    "bootstrap_run": False,
}
summary_path = OUT / "panelA_cedi_declipped_observed_gof_sensitivity_v2.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=== AUDIT COMPLETE ===")
print(f"detail:  {detail_path.relative_to(ROOT)}")
print(f"summary: {summary_path.relative_to(ROOT)}")
print("No bootstrap was run.")
