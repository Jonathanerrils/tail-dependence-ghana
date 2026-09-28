"""Diagnose the exact Panel A Cedi PIT tie found by script 33.

Diagnostic only:
- no model fitting
- no bootstrap
- no files modified outside outputs/pit_ties_audit/

Run from repository root:
    python scripts/34_panelA_cedi_pit_tie_diagnostic.py
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIT = ROOT / "outputs" / "tables" / "_pit_real.csv"
OUT = ROOT / "outputs" / "pit_ties_audit"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA = "be42455c12bf2f4a9c7ef33494930ad396f569192f571386d099046a388119d1"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not PIT.exists():
    raise FileNotFoundError(PIT)

pit = pd.read_csv(PIT, index_col=0, parse_dates=True).sort_index()
phash = sha256(PIT)

if "cedi" not in pit.columns:
    raise RuntimeError("Expected column 'cedi' not found.")

s = pit["cedi"]
vc = s.value_counts(dropna=False)
dup_vals = vc[vc > 1]

print("=== PANEL A CEDI PIT TIE DIAGNOSTIC ===")
print(f"PIT: {PIT.relative_to(ROOT)}")
print(f"SHA256: {phash}")
print(f"freeze hash match: {phash == EXPECTED_SHA}")
print(f"shape: {pit.shape}")
print(f"period: {pit.index.min().date()} to {pit.index.max().date()}")
print(f"duplicated PIT values: {len(dup_vals)}")
print()

rows = []
for val, mult in dup_vals.items():
    dates = s.index[s == val]
    less = int((s < val).sum())
    less_equal = int((s <= val).sum())
    avg_rank = (less + 1 + less_equal) / 2.0
    pct = avg_rank / (len(s) + 1.0)

    distinct = np.sort(s.drop_duplicates().to_numpy(dtype=float))
    pos = int(np.searchsorted(distinct, val))
    prev_val = float(distinct[pos - 1]) if pos > 0 else np.nan
    next_val = float(distinct[pos + 1]) if pos + 1 < len(distinct) else np.nan
    gap_prev = float(val - prev_val) if np.isfinite(prev_val) else np.nan
    gap_next = float(next_val - val) if np.isfinite(next_val) else np.nan

    lower_tail = bool(val <= 0.10)
    upper_tail = bool(val >= 0.90)

    print(f"tied PIT value: {val:.17g}")
    print(f"multiplicity: {int(mult)}")
    print("dates:")
    for d in dates:
        print(f"  {d.date()}")
    print(f"average empirical rank percentile: {pct:.6f}")
    print(f"in q<=0.10 lower tail: {lower_tail}")
    print(f"in q>=0.90 upper tail: {upper_tail}")
    print(f"previous distinct PIT: {prev_val:.17g}" if np.isfinite(prev_val) else "previous distinct PIT: none")
    print(f"next distinct PIT:     {next_val:.17g}" if np.isfinite(next_val) else "next distinct PIT: none")
    print(f"gap to previous: {gap_prev:.6g}" if np.isfinite(gap_prev) else "gap to previous: n/a")
    print(f"gap to next:     {gap_next:.6g}" if np.isfinite(gap_next) else "gap to next: n/a")
    print()

    for d in dates:
        rows.append({
            "pit_value": float(val),
            "multiplicity": int(mult),
            "date": str(d.date()),
            "average_rank_percentile": float(pct),
            "lower_tail_q10": lower_tail,
            "upper_tail_q90": upper_tail,
            "previous_distinct_pit": prev_val,
            "next_distinct_pit": next_val,
            "gap_to_previous": gap_prev,
            "gap_to_next": gap_next,
        })

detail = pd.DataFrame(rows)
detail_path = OUT / "panelA_cedi_tie_detail.csv"
detail.to_csv(detail_path, index=False)

summary = {
    "pit_sha256": phash,
    "freeze_hash_match": phash == EXPECTED_SHA,
    "n": int(len(s)),
    "n_unique": int(s.nunique()),
    "duplicate_count": int(len(s) - s.nunique()),
    "tied_groups": int(len(dup_vals)),
    "observations_in_tied_groups": int(dup_vals.sum()) if len(dup_vals) else 0,
    "share_observations_in_tied_groups": float(dup_vals.sum() / len(s)) if len(dup_vals) else 0.0,
    "max_multiplicity": int(vc.max()),
    "tied_values": [
        {
            "value": float(val),
            "multiplicity": int(mult),
            "dates": [str(d.date()) for d in s.index[s == val]],
            "is_lower_q10": bool(val <= 0.10),
            "is_upper_q90": bool(val >= 0.90),
        }
        for val, mult in dup_vals.items()
    ],
}
summary_path = OUT / "panelA_cedi_tie_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Decision note:")
print("- This script does not decide that the GOF is invalid.")
print("- If the tie is central and isolated, its direct numerical footprint is very small.")
print("- If it lies in an extreme PIT region, or comes from boundary clipping, it deserves closer scrutiny.")
print("- Because one Panel A Cocoa-Cedi Frank GOF p-value was near 0.05,")
print("  the exact tie location should be known before deciding whether a tie-adapted sensitivity is needed.")
print()
print(f"detail:  {detail_path.relative_to(ROOT)}")
print(f"summary: {summary_path.relative_to(ROOT)}")
print("=== DIAGNOSTIC COMPLETE ===")
