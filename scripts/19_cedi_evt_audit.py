"""Audit the Cedi EVT discrepancy without refitting any marginal model.

Purpose
-------
This script treats the already-frozen standardized residual file
`outputs/tables/_resid_real.csv` as the primary input and answers four
questions:

1. What does the current EVT implementation produce for the Cedi at the
   paper's q=0.90 POT threshold?
2. Does that exactly reproduce the current `evt_pot_gpd.csv` Cedi row?
3. How sensitive is the Cedi GPD fit to the POT threshold?
4. Do raw Cedi returns produce materially different EVT estimates, which
   may help explain legacy/ad-hoc outputs?

No existing publication output is overwritten. The script writes only
new audit files.

Run from the repository root:
    python scripts/19_cedi_evt_audit.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk import evt  # noqa: E402


TAB = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"

RESID_PATH = TAB / "_resid_real.csv"
RETURNS_PATH = PROC / "returns_real.csv"
CANONICAL_EVT_PATH = TAB / "evt_pot_gpd.csv"

AUDIT_SUMMARY_PATH = TAB / "cedi_evt_audit_current.csv"
AUDIT_SENS_PATH = TAB / "cedi_evt_threshold_sensitivity_audit.csv"
AUDIT_ALL_PATH = TAB / "evt_pot_gpd_recomputed_audit.csv"
AUDIT_INVENTORY_PATH = TAB / "cedi_evt_existing_file_inventory.csv"

Q0 = 0.90
ALPHA = 0.99
THRESHOLDS = np.arange(0.85, 0.976, 0.005)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_summary(losses: pd.Series | np.ndarray, source: str, q: float = Q0) -> dict:
    x = pd.Series(losses).dropna().astype(float)
    pot = evt.fit_pot(x, quantile=q)
    k = max(25, int(0.05 * len(x)))
    return {
        "source": source,
        "quantile": q,
        "n_total": int(len(x)),
        "threshold": pot.threshold,
        "xi": pot.xi,
        "beta": pot.beta,
        "n_exceed": pot.n_exceed,
        "VaR99": pot.var(ALPHA),
        "ES99": pot.es(ALPHA),
        "hill_gamma_k5pct": evt.hill_estimator(x, k),
        "hill_k": k,
    }


def direct_scipy_check(losses: pd.Series | np.ndarray, q: float = Q0) -> dict:
    x = pd.Series(losses).dropna().astype(float).to_numpy()
    u = float(np.quantile(x, q))
    exc = x[x > u] - u
    xi, loc, beta = stats.genpareto.fit(exc, floc=0.0)

    zeta = len(exc) / len(x)
    var99 = u + (beta / xi) * (((1 - ALPHA) / zeta) ** (-xi) - 1)
    es99 = var99 / (1 - xi) + (beta - xi * u) / (1 - xi)

    return {
        "threshold": float(u),
        "xi": float(xi),
        "loc": float(loc),
        "beta": float(beta),
        "n_exceed": int(len(exc)),
        "VaR99": float(var99),
        "ES99": float(es99),
    }


def read_existing_evt_files() -> pd.DataFrame:
    rows = []
    for path in sorted(TAB.glob("*evt*.csv")):
        row = {
            "file": path.name,
            "sha256": sha256(path),
            "readable": False,
            "has_cedi_row": False,
            "cedi_row": "",
        }
        try:
            df = pd.read_csv(path)
            row["readable"] = True
            if "series" in df.columns:
                mask = df["series"].astype(str).str.lower().eq("cedi")
                if mask.any():
                    row["has_cedi_row"] = True
                    row["cedi_row"] = df.loc[mask].iloc[0].to_json()
        except Exception as exc:
            row["cedi_row"] = f"READ ERROR: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


if not RESID_PATH.exists():
    raise FileNotFoundError(f"Missing frozen residual input: {RESID_PATH}")

resid = pd.read_csv(RESID_PATH, index_col=0, parse_dates=True)

if "cedi" not in resid.columns:
    raise KeyError(f"'cedi' column not found in {RESID_PATH}")

cedi_resid_losses = -resid["cedi"].dropna()

print("=== INPUT FREEZE ===")
print(f"_resid_real.csv SHA256: {sha256(RESID_PATH)}")
print(f"evt.py SHA256:           {sha256(ROOT / 'src' / 'tailrisk' / 'evt.py')}")
print(f"Residual rows:           {len(cedi_resid_losses)}")
print(f"Residual date range:     {cedi_resid_losses.index.min().date()} to "
      f"{cedi_resid_losses.index.max().date()}")

resid_result = fit_summary(
    cedi_resid_losses,
    source="negative standardized residuals (_resid_real.csv)",
)
direct = direct_scipy_check(cedi_resid_losses)

print("\n=== CURRENT CEDI POT/GPD FROM FROZEN STANDARDIZED RESIDUALS ===")
for key in ["quantile", "n_total", "threshold", "xi", "beta",
            "n_exceed", "VaR99", "ES99", "hill_gamma_k5pct", "hill_k"]:
    value = resid_result[key]
    if isinstance(value, float):
        print(f"{key:20s}: {value:.12f}")
    else:
        print(f"{key:20s}: {value}")

print("\n=== DIRECT SCIPY CROSS-CHECK ===")
for key, value in direct.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.12f}")
    else:
        print(f"{key:20s}: {value}")

assert np.isclose(resid_result["threshold"], direct["threshold"], rtol=0, atol=1e-12)
assert np.isclose(resid_result["xi"], direct["xi"], rtol=0, atol=1e-12)
assert np.isclose(resid_result["beta"], direct["beta"], rtol=0, atol=1e-12)
assert resid_result["n_exceed"] == direct["n_exceed"]
assert np.isclose(resid_result["VaR99"], direct["VaR99"], rtol=0, atol=1e-12)
assert np.isclose(resid_result["ES99"], direct["ES99"], rtol=0, atol=1e-12)
print("Direct SciPy cross-check: PASSED")

comparison_status = "MISSING"
legacy_row = None

if CANONICAL_EVT_PATH.exists():
    old = pd.read_csv(CANONICAL_EVT_PATH)
    if "series" in old.columns:
        cedi_old = old[old["series"].astype(str).str.lower().eq("cedi")]
        if len(cedi_old) == 1:
            legacy_row = cedi_old.iloc[0]
            mapping = {
                "u(q90)": "threshold",
                "xi": "xi",
                "beta": "beta",
                "n_exceed": "n_exceed",
                "VaR99_resid": "VaR99",
                "ES99_resid": "ES99",
                "hill_gamma(k=5%)": "hill_gamma_k5pct",
            }
            print("\n=== EXISTING evt_pot_gpd.csv VS CURRENT RECOMPUTATION ===")
            all_match_rounded = True
            for old_col, new_key in mapping.items():
                if old_col not in old.columns:
                    print(f"{old_col:22s}: column missing in existing file")
                    all_match_rounded = False
                    continue
                old_val = float(legacy_row[old_col])
                new_val = float(resid_result[new_key])
                delta = new_val - old_val
                match3 = round(new_val, 3) == round(old_val, 3)
                all_match_rounded &= match3
                print(
                    f"{old_col:22s}: existing={old_val:.12f}  "
                    f"current={new_val:.12f}  delta={delta:+.12f}  "
                    f"match_3dp={match3}"
                )
            comparison_status = (
                "MATCHES_EXISTING_TO_3DP"
                if all_match_rounded
                else "DIFFERS_FROM_EXISTING"
            )
            print(f"Comparison status: {comparison_status}")
        else:
            comparison_status = f"CEDI_ROWS_FOUND_{len(cedi_old)}"

print(f"\nCanonical comparison: {comparison_status}")

all_rows = []
for col in resid.columns:
    losses = -resid[col].dropna()
    all_rows.append(fit_summary(losses, source=col))

all_evt = pd.DataFrame(all_rows)
all_evt.to_csv(
    AUDIT_ALL_PATH,
    index=False,
    float_format="%.12f",
    lineterminator="\n",
    encoding="utf-8",
)
print(f"\nWrote: {AUDIT_ALL_PATH.relative_to(ROOT)}")

sens_rows = []
for q in THRESHOLDS:
    r = fit_summary(
        cedi_resid_losses,
        source="negative standardized residuals (_resid_real.csv)",
        q=float(q),
    )
    sens_rows.append(r)

sens = pd.DataFrame(sens_rows)
sens.to_csv(
    AUDIT_SENS_PATH,
    index=False,
    float_format="%.12f",
    lineterminator="\n",
    encoding="utf-8",
)
print(f"Wrote: {AUDIT_SENS_PATH.relative_to(ROOT)}")

print("\n=== CEDI THRESHOLD SENSITIVITY ===")
print(
    sens[["quantile", "n_exceed", "threshold", "xi", "beta", "VaR99", "ES99"]]
    .round(6)
    .to_string(index=False)
)

raw_result = None
if RETURNS_PATH.exists():
    returns = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    if "cedi" in returns.columns:
        raw_losses = -returns["cedi"].dropna()
        raw_result = fit_summary(
            raw_losses,
            source="negative raw Cedi returns (returns_real.csv)",
        )
        print("\n=== RAW CEDI RETURN POT/GPD DIAGNOSTIC ===")
        for key in ["n_total", "threshold", "xi", "beta", "n_exceed",
                    "VaR99", "ES99", "hill_gamma_k5pct"]:
            value = raw_result[key]
            if isinstance(value, float):
                print(f"{key:20s}: {value:.12f}")
            else:
                print(f"{key:20s}: {value}")

inventory = read_existing_evt_files()
inventory.to_csv(
    AUDIT_INVENTORY_PATH,
    index=False,
    lineterminator="\n",
    encoding="utf-8",
)
print(f"\nWrote: {AUDIT_INVENTORY_PATH.relative_to(ROOT)}")

if len(inventory):
    print("\n=== EXISTING EVT-NAMED CSV FILES ===")
    print(inventory[["file", "has_cedi_row", "cedi_row"]].to_string(index=False))
else:
    print("\nNo existing *evt*.csv files found in outputs/tables.")

summary_rows = [resid_result]
if raw_result is not None:
    summary_rows.append(raw_result)

summary = pd.DataFrame(summary_rows)
summary["resid_input_sha256"] = sha256(RESID_PATH)
summary["evt_code_sha256"] = sha256(ROOT / "src" / "tailrisk" / "evt.py")
summary["canonical_comparison"] = comparison_status

summary.to_csv(
    AUDIT_SUMMARY_PATH,
    index=False,
    float_format="%.12f",
    lineterminator="\n",
    encoding="utf-8",
)
print(f"Wrote: {AUDIT_SUMMARY_PATH.relative_to(ROOT)}")

print("\n=== AUDIT COMPLETE ===")
print("No existing EVT publication file was overwritten.")
print("Interpret the standardized-residual result as the canonical pipeline definition.")
print("Treat the raw-return result only as a source-tracing diagnostic.")
