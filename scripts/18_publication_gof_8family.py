"""Publication-resolution copula GOF sweep, two-phase design:

Phase A: all 10 pairs x 8 families at B=2,000, checkpointed after every
single (pair, family) test. Safe to interrupt and resume at any point --
on restart, already-completed tests at the target resolution are skipped
entirely, not recomputed.

Phase B: any cell landing in the pre-declared borderline zone
0.03 <= p <= 0.07 at B=2,000 is automatically rerun at B=5,000. The
higher-resolution result replaces the B=2,000 result for that cell in
the final table. Escalation is triggered mechanically by the interval,
never by which result would be more convenient.

Why 2,000, not 5,000, uniformly: the estimator-unification fix (using
copulas.FAMILIES' pseudo-MLE throughout, not a separate closed-form
approximation) was the real methodological correction here. Bootstrap
resolution is secondary. At p=0.05, the Monte Carlo standard error is
already close to 0.005 at B=2,000, materially better than B=1,000
(~0.007), while B=5,000 for cells nowhere near the 0.05 boundary buys
little for a large time cost. Escalating only genuinely borderline
cells spends the expensive resolution where it can actually change a
decision.
"""

import sys, itertools, hashlib, time, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from tailrisk import copulas, gof

def stable_seed(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

B_BASE = 2_000
B_ESCALATE = 5_000
BORDERLINE_LO, BORDERLINE_HI = 0.03, 0.07
FAMILIES = ["gaussian", "t", "clayton", "gumbel", "frank", "joe",
            "survival_clayton", "survival_gumbel"]

CHECKPOINT_PATH = "outputs/tables/copula_gof_8family_checkpoint.csv"
FINAL_PATH = "outputs/tables/copula_gof_8family_B2000_B5000.csv"

pit = pd.read_csv("outputs/tables/_pit_real.csv", index_col=0, parse_dates=True)
LABELS = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI", "cedi": "Cedi"}
pairs = list(itertools.combinations(pit.columns, 2))
all_cells = [(a, b, fam) for a, b in pairs for fam in FAMILIES]

print(f"Phase A: {len(all_cells)} (pair, family) tests at B={B_BASE}")
print(f"Phase B: automatic escalation to B={B_ESCALATE} for p in "
      f"[{BORDERLINE_LO}, {BORDERLINE_HI}]\n")

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = pd.read_csv(CHECKPOINT_PATH)
    print(f"Resuming from existing checkpoint: {len(checkpoint)} rows already present.")
    if len(checkpoint):
        assert not checkpoint.duplicated(["pair_a", "pair_b", "family", "n_boot"]).any(), \
            "Duplicate checkpoint cells found -- checkpoint may be corrupted"
        assert set(checkpoint["family"]).issubset(set(FAMILIES)), \
            f"Checkpoint contains unknown families: {set(checkpoint['family']) - set(FAMILIES)}"
        assert set(checkpoint["n_boot"]).issubset({B_BASE, B_ESCALATE}), \
            f"Checkpoint contains unexpected n_boot values: {set(checkpoint['n_boot']) - {B_BASE, B_ESCALATE}}"
else:
    checkpoint = pd.DataFrame(columns=["pair_a", "pair_b", "family", "n_boot",
                                       "Sn", "p_value", "rho", "nu", "theta"])
    print("No existing checkpoint found, starting fresh.")


def already_done(a, b, fam, n_boot):
    if len(checkpoint) == 0:
        return False
    m = ((checkpoint["pair_a"] == a) & (checkpoint["pair_b"] == b) &
         (checkpoint["family"] == fam) & (checkpoint["n_boot"] == n_boot))
    return m.any()


def append_result(a, b, fam, n_boot, r):
    global checkpoint
    row = {"pair_a": a, "pair_b": b, "family": fam, "n_boot": n_boot,
           "Sn": r["Sn"], "p_value": r["p_value"],
           "rho": r["params"].get("rho", np.nan),
           "nu": r["params"].get("nu", np.nan),
           "theta": r["params"].get("theta", np.nan)}
    checkpoint = pd.concat([checkpoint, pd.DataFrame([row])], ignore_index=True)
    # Atomic write: a shutdown mid-write can leave a truncated file if
    # writing directly to CHECKPOINT_PATH. Writing to a temp file first
    # and swapping it in with os.replace() (atomic on both POSIX and
    # Windows) means the checkpoint is always either the previous
    # complete version or the new complete version, never a partial one.
    tmp = CHECKPOINT_PATH + ".tmp"
    checkpoint.to_csv(tmp, index=False, float_format="%.6f",
                      lineterminator="\n", encoding="utf-8")
    os.replace(tmp, CHECKPOINT_PATH)


pit_cache = {}
def get_uv(a, b):
    if (a, b) not in pit_cache:
        u = copulas.pseudo_obs(pit[[a]])[:, 0]
        v = copulas.pseudo_obs(pit[[b]])[:, 0]
        pit_cache[(a, b)] = (u, v)
    return pit_cache[(a, b)]


# ------------------------------------------------------------- Phase A
t0 = time.time()
n_run = 0
for i, (a, b, fam) in enumerate(all_cells):
    if already_done(a, b, fam, B_BASE):
        continue
    u, v = get_uv(a, b)
    r = gof.gof_test(u, v, fam, n_boot=B_BASE, seed=stable_seed(a, b, fam, B_BASE))
    append_result(a, b, fam, B_BASE, r)
    n_run += 1
    print(f"[Phase A {i+1}/{len(all_cells)}] {LABELS[a]}-{LABELS[b]} {fam:20s} "
          f"p={r['p_value']:.4f}  ({time.time()-t0:.1f}s elapsed this run, "
          f"{n_run} tests run this session)", flush=True)

print(f"\nPhase A complete: all {len(all_cells)} tests present at B={B_BASE}.")

base = checkpoint[checkpoint["n_boot"] == B_BASE]
assert len(base) == 80, \
    f"Expected exactly 80 B={B_BASE} cells after Phase A, got {len(base)}"
assert not base.duplicated(["pair_a", "pair_b", "family"]).any(), \
    "Duplicate pair/family cells found in B_BASE checkpoint after Phase A"

# ------------------------------------------------------------- Phase B
borderline = base[(base["p_value"] >= BORDERLINE_LO) & (base["p_value"] <= BORDERLINE_HI)]
print(f"\nBorderline cells (p in [{BORDERLINE_LO}, {BORDERLINE_HI}]) at B={B_BASE}: "
      f"{len(borderline)}")
if len(borderline):
    print(borderline[["pair_a", "pair_b", "family", "p_value"]].to_string(index=False))

for _, row in borderline.iterrows():
    a, b, fam = row["pair_a"], row["pair_b"], row["family"]
    if already_done(a, b, fam, B_ESCALATE):
        continue
    u, v = get_uv(a, b)
    r = gof.gof_test(u, v, fam, n_boot=B_ESCALATE, seed=stable_seed(a, b, fam, B_ESCALATE))
    append_result(a, b, fam, B_ESCALATE, r)
    print(f"[Phase B escalation] {LABELS[a]}-{LABELS[b]} {fam:20s} "
          f"B={B_BASE}->p={row['p_value']:.4f}  B={B_ESCALATE}->p={r['p_value']:.4f}  "
          f"({time.time()-t0:.1f}s elapsed)", flush=True)

# ------------------------------------------------------------- Final table
final_rows = []
for a, b in pairs:
    for fam in FAMILIES:
        cell = checkpoint[(checkpoint["pair_a"] == a) & (checkpoint["pair_b"] == b) &
                          (checkpoint["family"] == fam)]
        base_row = cell[cell["n_boot"] == B_BASE].iloc[0]
        escalated = cell[cell["n_boot"] == B_ESCALATE]
        was_escalated = len(escalated) > 0
        chosen = escalated.iloc[0] if was_escalated else base_row
        final_rows.append({
            "pair": f"{LABELS[a]}\u2013{LABELS[b]}", "family": fam,
            "n_boot_used": int(chosen["n_boot"]),
            "escalated": was_escalated,
            "p_value_B2000": base_row["p_value"],
            "p_value_B5000": escalated.iloc[0]["p_value"] if was_escalated else np.nan,
            "p_value_final": chosen["p_value"],
            "Sn": chosen["Sn"],
            "not_rejected_5pct": chosen["p_value"] >= 0.05,
            "rho": chosen["rho"], "nu": chosen["nu"], "theta": chosen["theta"],
        })

final = pd.DataFrame(final_rows)

assert len(final) == 80, f"Expected 80 rows, got {len(final)}"
assert final["pair"].nunique() == 10
assert set(final["family"]) == set(FAMILIES)
assert not final.duplicated(["pair", "family"]).any()
assert final["p_value_final"].between(0, 1).all()
assert (final["Sn"] >= 0).all()

final.round(6).sort_values(["pair", "family"]).to_csv(
    FINAL_PATH, index=False, float_format="%.6f",
    lineterminator="\n", encoding="utf-8",
)

print(f"\nTotal time this session: {time.time()-t0:.1f}s")
print(f"Escalated cells: {int(final['escalated'].sum())}")
print("\n=== Families NOT rejected (p >= 0.05), per pair ===")
print(final.groupby("pair")["not_rejected_5pct"].sum().sort_values().to_string())
print(f"\nFinal table written to {FINAL_PATH}")
print("Structural assertions: PASSED")