"""Panel B publication-resolution stress inference, safety/checkpoint edition.

Preflight only:
    python scripts/31_publication_rebuild_stage3_stress_SAFE.py --preflight

Run one 60-test family:
    python scripts/31_publication_rebuild_stage3_stress_SAFE.py --family combined
    python scripts/31_publication_rebuild_stage3_stress_SAFE.py --family covid
    python scripts/31_publication_rebuild_stage3_stress_SAFE.py --family 2024

Run all three sequentially only when explicitly intended:
    python scripts/31_publication_rebuild_stage3_stress_SAFE.py --family all

The script checkpoints after every completed test and resumes safely.
It does not overwrite Panel A outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "publication_rebuild_v2" / "tables"
PITFILE = V2 / "_pit_publication_v2.csv"

EXPECTED_PIT_SHA = "9b322bda7c4e63534da14d1a7e360e758762e98cb39101cf329cdceb5afc216f"
EXPECTED_SHAPE = (2773, 5)
EXPECTED_START = "2015-01-06"
EXPECTED_END = "2026-07-10"

B = 15000
BLOCK = 20
ALPHA = 0.05
QS = (0.025, 0.05, 0.10)
FAMILIES = ("combined", "covid", "2024")

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-06-30")
Y2024_START = pd.Timestamp("2024-01-01")
Y2024_END = pd.Timestamp("2024-12-31")

LABELS = {"cocoa":"Cocoa", "gold":"Gold", "brent":"Brent", "wti":"WTI", "cedi":"Cedi"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    raw = "||".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def rerank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x) / (len(x) + 1.0)


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(2, min(block, n // 4))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def empirical_lambda(u: np.ndarray, v: np.ndarray, q: float, upper: bool) -> float:
    if upper:
        return float(np.mean((u >= 1.0 - q) & (v >= 1.0 - q)) / q)
    return float(np.mean((u <= q) & (v <= q)) / q)


def one_test(cu0, cv0, su0, sv0, q: float, upper: bool, seed: int) -> dict:
    # Mirrors the publication-resolution logic: rerank within regime,
    # then rerank every moving-block bootstrap replicate.
    cu, cv = rerank(cu0), rerank(cv0)
    su, sv = rerank(su0), rerank(sv0)

    lc = empirical_lambda(cu, cv, q, upper)
    ls = empirical_lambda(su, sv, q, upper)
    observed = ls - lc

    rng = np.random.default_rng(seed)
    diff = np.empty(B, dtype=float)

    for i in range(B):
        ic = block_indices(len(cu), BLOCK, rng)
        is_ = block_indices(len(su), BLOCK, rng)

        cub, cvb = rerank(cu[ic]), rerank(cv[ic])
        sub, svb = rerank(su[is_]), rerank(sv[is_])

        diff[i] = (
            empirical_lambda(sub, svb, q, upper)
            - empirical_lambda(cub, cvb, q, upper)
        )

    lo, hi = np.quantile(diff, [0.025, 0.975])
    p_value = float(
        (1 + np.sum(np.abs(diff - observed) >= abs(observed))) / (B + 1)
    )

    return {
        "lambda_calm": lc,
        "lambda_stress": ls,
        "difference": observed,
        "diff_lo_95": float(lo),
        "diff_hi_95": float(hi),
        "p_value": p_value,
    }


def bh_flags(pvals: np.ndarray, alpha: float = 0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = np.where(ranked <= thresholds)[0]
    flags = np.zeros(m, dtype=bool)
    cutoff = None
    if len(passing):
        cutoff = float(ranked[passing.max()])
        flags = pvals <= cutoff
    return flags, cutoff


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g")
    os.replace(tmp, path)


def regime_masks(pit: pd.DataFrame):
    covid = pd.Series((pit.index >= COVID_START) & (pit.index <= COVID_END), index=pit.index)
    y24 = pd.Series((pit.index >= Y2024_START) & (pit.index <= Y2024_END), index=pit.index)
    calm = ~(covid | y24)
    return calm, covid, y24


def family_mask(name: str, covid: pd.Series, y24: pd.Series):
    if name == "combined":
        return covid | y24
    if name == "covid":
        return covid
    if name == "2024":
        return y24
    raise ValueError(name)


def output_paths(name: str):
    return (
        V2 / f"stress_{name}_B15000_checkpoint_v2.csv",
        V2 / f"stress_{name}_B15000_final_v2.csv",
        V2 / f"stress_{name}_B15000_summary_v2.json",
    )


def row_key(r) -> tuple:
    return (str(r["a"]), str(r["b"]), str(r["tail"]), float(r["q"]))


def run_family(pit: pd.DataFrame, name: str):
    calm_mask, covid_mask, y24_mask = regime_masks(pit)
    stress_mask = family_mask(name, covid_mask, y24_mask)
    calm, stress = pit.loc[calm_mask], pit.loc[stress_mask]

    checkpoint, final_path, summary_path = output_paths(name)

    if checkpoint.exists():
        cp = pd.read_csv(checkpoint)
        rows = cp.to_dict("records")
        done = {row_key(r) for _, r in cp.iterrows()}
        print(f"Resuming {name}: {len(done)}/60 tests complete.")
    else:
        rows, done = [], set()

    print(f"\n=== {name.upper()} ===")
    print(f"calm n={len(calm)}, stress n={len(stress)}, B={B}, block={BLOCK}")
    print(f"checkpoint: {checkpoint.relative_to(ROOT)}")

    started = time.time()
    pairs = list(itertools.combinations(pit.columns, 2))

    for a, b in pairs:
        for q in QS:
            for upper in (False, True):
                tail = "upper" if upper else "lower"
                key = (a, b, tail, float(q))
                if key in done:
                    continue

                seed = stable_seed("publication_v2_stress", name, a, b, q, tail, B, BLOCK)
                t0 = time.time()
                result = one_test(
                    calm[a].to_numpy(), calm[b].to_numpy(),
                    stress[a].to_numpy(), stress[b].to_numpy(),
                    q, upper, seed,
                )

                rows.append({
                    "stress_definition": name,
                    "pair": f"{LABELS[a]}–{LABELS[b]}",
                    "a": a, "b": b, "tail": tail, "q": q,
                    "n_calm": len(calm), "n_stress": len(stress),
                    "B": B, "block": BLOCK, "seed": seed,
                    **result,
                })

                df = pd.DataFrame(rows).sort_values(["a", "b", "q", "tail"]).reset_index(drop=True)
                atomic_csv(df, checkpoint)
                print(
                    f"[{len(df):02d}/60] {LABELS[a]}–{LABELS[b]:<12s} "
                    f"{tail:5s} q={q:.3f} diff={result['difference']:+.6f} "
                    f"p={result['p_value']:.6f} ({time.time()-t0:.1f}s)",
                    flush=True,
                )

    df = pd.read_csv(checkpoint)
    if len(df) != 60:
        raise RuntimeError(f"{name} checkpoint incomplete: {len(df)}/60")

    p = df["p_value"].to_numpy(dtype=float)
    df["uncorrected_reject_5pct"] = p <= ALPHA
    df["bonf_threshold"] = ALPHA / 60
    df["bonf_reject_5pct"] = p <= (ALPHA / 60)
    bh, cutoff = bh_flags(p, ALPHA)
    df["bh_reject_5pct"] = bh
    df["bh_cutoff"] = np.nan if cutoff is None else cutoff
    df.to_csv(final_path, index=False, float_format="%.12g")

    cedi = (df["a"] == "cedi") | (df["b"] == "cedi")
    corrected = df["bonf_reject_5pct"] | df["bh_reject_5pct"]

    summary = {
        "stress_definition": name,
        "pit_sha256": sha256(PITFILE),
        "n_calm": int(len(calm)),
        "n_stress": int(len(stress)),
        "n_tests": 60,
        "B": B,
        "block": BLOCK,
        "min_p": float(p.min()),
        "uncorrected_rejections": int(df["uncorrected_reject_5pct"].sum()),
        "bonf_rejections": int(df["bonf_reject_5pct"].sum()),
        "bh_rejections": int(df["bh_reject_5pct"].sum()),
        "bonf_threshold": float(ALPHA / 60),
        "bh_cutoff": None if cutoff is None else float(cutoff),
        "corrected_cedi_rejections": int((corrected & cedi).sum()),
        "cedi_A_to_G_triggered": bool((corrected & cedi).any()),
        "runtime_seconds_this_invocation": float(time.time() - started),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== {name.upper()} SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if corrected.any():
        print("\nCorrected discoveries:")
        print(df.loc[corrected, ["pair", "tail", "q", "difference", "p_value", "bonf_reject_5pct", "bh_reject_5pct"]]
              .sort_values("p_value").to_string(index=False))
    else:
        print("\nCorrected discoveries: none")

    return final_path, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--family", choices=("combined", "covid", "2024", "all"))
    args = parser.parse_args()

    if not PITFILE.exists():
        raise FileNotFoundError(PITFILE)

    pit_sha = sha256(PITFILE)
    pit = pd.read_csv(PITFILE, index_col=0, parse_dates=True).sort_index()

    print("=== PANEL B STRESS PREFLIGHT ===")
    print(f"PIT: {PITFILE.relative_to(ROOT)}")
    print(f"SHA256: {pit_sha}")
    print(f"shape: {pit.shape}")
    print(f"period: {pit.index.min().date()} to {pit.index.max().date()}")

    checks = [
        pit_sha == EXPECTED_PIT_SHA,
        tuple(pit.shape) == EXPECTED_SHAPE,
        str(pit.index.min().date()) == EXPECTED_START,
        str(pit.index.max().date()) == EXPECTED_END,
    ]
    if not all(checks):
        raise SystemExit("Freeze check failed. Refusing to run expensive inference.")

    calm, covid, y24 = regime_masks(pit)
    print("freeze check: PASS")
    print(f"calm={int(calm.sum())}, COVID={int(covid.sum())}, 2024={int(y24.sum())}, combined={int((covid|y24).sum())}")
    print(f"design: 60 tests/family, B={B}, block={BLOCK}, Bonferroni={ALPHA/60:.12g}")

    if args.preflight:
        print("Preflight only. No bootstrap computation was run.")
        return

    if args.family is None:
        raise SystemExit("No expensive analysis started. Pass --family combined, covid, 2024, or all explicitly.")

    selected = FAMILIES if args.family == "all" else (args.family,)
    output_files = []
    for name in selected:
        output_files.extend(run_family(pit, name))

    hash_path = V2 / "stage3_stress_hashes_v2.txt"
    with hash_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha256(PITFILE)}  {PITFILE.relative_to(ROOT)}\n")
        for p in output_files:
            f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")

    print("\n=== OUTPUT HASHES ===")
    print(hash_path.read_text(encoding="utf-8"))
    print("=== PANEL B STRESS REQUEST COMPLETE ===")


if __name__ == "__main__":
    main()
