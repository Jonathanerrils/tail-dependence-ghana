from pathlib import Path
import hashlib, inspect, json, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tailrisk import copulas

PIT = ROOT / "outputs" / "tables" / "_pit_real.csv"
OUT = ROOT / "outputs" / "pit_ties_audit"
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_SHA = "be42455c12bf2f4a9c7ef33494930ad396f569192f571386d099046a388119d1"
FLOOR = 1e-6
QS = [0.025, 0.05, 0.10]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def scan_python_sources():
    hits = []
    for base in (ROOT / "src", ROOT / "scripts"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(lines, 1):
                low = line.lower()
                explicit_floor = any(t in low for t in ("1e-6", "1.e-6", "0.000001"))
                pit_clip = "clip" in low and any(t in low for t in ("pit", "cdf", "prob", "uniform"))
                if explicit_floor or pit_clip:
                    hits.append({"file": str(path.relative_to(ROOT)), "line": lineno, "text": line.strip()})
    return hits

pit = pd.read_csv(PIT, index_col=0, parse_dates=True).sort_index()
phash = sha256(PIT)
cedi = pit["cedi"].astype(float)
floor_mask = cedi.to_numpy() == FLOOR
floor_dates = cedi.index[floor_mask]
floor_count = int(floor_mask.sum())

print("=== PANEL A PIT FLOOR AUDIT (SAFE) ===")
print("This version reads only the Panel A PIT CSV and Python source text.")
print(f"PIT: {PIT.relative_to(ROOT)}")
print(f"SHA256: {phash}")
print(f"freeze hash match: {phash == EXPECTED_SHA}")
print(f"shape: {pit.shape}")
print(f"period: {pit.index.min().date()} to {pit.index.max().date()}")
print()
print("--- Exact 1e-6 Cedi PIT observations ---")
print(f"count: {floor_count}")
for d in floor_dates:
    print(f"  {d.date()}  PIT={cedi.loc[d]:.17g}")
print()
print("--- Local pseudo_obs implementation ---")
try:
    pseudo_src = inspect.getsource(copulas.pseudo_obs)
except Exception as exc:
    pseudo_src = f"<inspect failed: {exc}>"
print(pseudo_src.rstrip())
print()

n = len(cedi)
avg_ranks = rankdata(cedi.to_numpy(), method="average")
avg_u = avg_ranks / (n + 1.0)
idx = np.where(floor_mask)[0]
current_ranks = avg_ranks[idx]
current_u = avg_u[idx]

print("--- Rank effect of the floor tie ---")
print(f"sample n: {n}")
print(f"current average ranks: {current_ranks.tolist()}")
print(f"current pseudo-u: {[float(x) for x in current_u]}")

possible_u = np.array([])
max_shift = None
if floor_count == 3:
    possible_ranks = np.array([1.0, 2.0, 3.0])
    possible_u = possible_ranks / (n + 1.0)
    current = float(current_u[0])
    max_shift = float(np.max(np.abs(possible_u - current)))
    print(f"possible distinct ranks: {possible_ranks.tolist()}")
    print(f"possible pseudo-u: {[float(x) for x in possible_u]}")
    print(f"maximum pseudo-u displacement: {max_shift:.12g}")
print()

print("--- Tail-threshold membership ---")
threshold_rows = []
for q in QS:
    a = bool(np.all(current_u <= q)) if floor_count else None
    b = bool(np.all(possible_u <= q)) if len(possible_u) else None
    print(f"q={q:.3f}: current all <= q? {a}; possible distinct ranks all <= q? {b}")
    threshold_rows.append({"q": q, "current_all_below_q": a, "possible_distinct_all_below_q": b})
print()

print("--- Source scan for clipping/floor logic ---")
hits = scan_python_sources()
if hits:
    for h in hits:
        print(f"{h['file']}:{h['line']}: {h['text']}")
else:
    print("No explicit floor/clipping line found under src/ or scripts/.")
print()

summary = {
    "pit_sha256": phash,
    "freeze_hash_match": phash == EXPECTED_SHA,
    "floor_value": FLOOR,
    "floor_count": floor_count,
    "floor_dates": [str(d.date()) for d in floor_dates],
    "current_average_ranks": [float(x) for x in current_ranks],
    "current_pseudo_u": [float(x) for x in current_u],
    "possible_distinct_pseudo_u": [float(x) for x in possible_u],
    "max_pseudo_u_displacement": max_shift,
    "threshold_membership": threshold_rows,
    "source_hits": hits,
    "pseudo_obs_source": pseudo_src,
}
summary_path = OUT / "panelA_cedi_pit_floor_audit_SAFE.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=== SAFE AUDIT INTERPRETATION ===")
if floor_count == 3:
    print("The three observations occupy the three smallest Cedi PIT ranks.")
    print("Breaking the tie can only assign ranks 1, 2 and 3.")
    print("Their q=.025, .05 and .10 lower-tail membership is unchanged.")
print("No GOF rerun is authorized by this script.")
print(f"summary: {summary_path.relative_to(ROOT)}")
print("=== SAFE PIT FLOOR AUDIT COMPLETE ===")
