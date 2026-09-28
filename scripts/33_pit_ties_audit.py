"""Audit exact ties in the PIT inputs used for copula GOF.

Diagnostic only. Performs no bootstrap.

Run from repository root:
    python scripts/33_pit_ties_audit.py
"""

from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANELS = {
    "A": ROOT / "outputs" / "tables" / "_pit_real.csv",
    "B": ROOT / "outputs" / "publication_rebuild_v2" / "tables" / "_pit_publication_v2.csv",
}
EXPECTED_B_SHA = "9b322bda7c4e63534da14d1a7e360e758762e98cb39101cf329cdceb5afc216f"
OUT = ROOT / "outputs" / "pit_ties_audit"
OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
summary = {}

for panel, path in PANELS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    phash = sha256(path)
    s = {
        "path": str(path.relative_to(ROOT)),
        "sha256": phash,
        "shape": list(df.shape),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
    }
    if panel == "B":
        s["freeze_check"] = phash == EXPECTED_B_SHA

    any_ties = False
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        n = len(df)
        n_unique = int(df[col].nunique(dropna=False))
        tied = vc[vc > 1]
        tied_groups = int(len(tied))
        tied_obs = int(tied.sum()) if tied_groups else 0
        max_mult = int(vc.max())
        rows.append({
            "panel": panel,
            "series": col,
            "n": n,
            "n_unique": n_unique,
            "duplicate_count": n - n_unique,
            "tied_groups": tied_groups,
            "observations_in_tied_groups": tied_obs,
            "share_observations_in_tied_groups": tied_obs / n,
            "max_multiplicity": max_mult,
        })
        any_ties |= tied_groups > 0

    s["any_exact_PIT_ties"] = any_ties
    summary[panel] = s

out = pd.DataFrame(rows)
out.to_csv(OUT / "pit_ties_by_series.csv", index=False)
(OUT / "pit_ties_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=== PIT TIES AUDIT ===")
for panel in ("A", "B"):
    s = summary[panel]
    print(f"Panel {panel}: {tuple(s['shape'])}, {s['start']} to {s['end']}")
    print(f"SHA256: {s['sha256']}")
    if panel == "B":
        print(f"freeze check: {s['freeze_check']}")
    sub = out[out["panel"] == panel]
    print(sub[[
        "series","n_unique","duplicate_count","tied_groups",
        "observations_in_tied_groups",
        "share_observations_in_tied_groups","max_multiplicity"
    ]].to_string(index=False))
    print()

print("Interpretation:")
print("- If all duplicate_count values are zero, the no-ties issue is closed for GOF inputs.")
print("- If ties exist, do not infer invalidity automatically.")
print("  Quantify where they occur before deciding whether a tie-adapted GOF sensitivity is warranted.")
print(f"outputs: {OUT.relative_to(ROOT)}")
