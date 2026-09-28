"""End-to-end tail-dependence pipeline.

Usage:
    python scripts/02_run_pipeline.py                       # synthetic demo
    python scripts/02_run_pipeline.py --data real           # real data
    python scripts/02_run_pipeline.py --data real \
        --ingest-dir /path/to/project2_tail_dependence_data

Real-data input
---------------
Consumes the output of the Project 2 data-ingestion script
(`daily_prices_wide.csv` in <ingest-dir>/data_processed/), i.e. the wide
daily price panel with columns such as cocoa_futures, gold_futures,
gold_spot_proxy, brent_spot_fred, wti_spot_fred, wti_futures,
ghana_usd_exchange_daily. The loader maps these to canonical names,
aligns calendars, cleans, recomputes percent log returns, and writes
canonical prices_real.csv / returns_real.csv under data/processed/ so
every downstream artifact (dashboard included) sees one schema.

Stages
------
1. Marginals: AR(1)-GJR-GARCH(1,1)-t per series; PIT residuals.
2. EVT: POT/GPD on loss tails; threshold sensitivity for cocoa; Hill.
3. Copulas: Gaussian/t/Clayton/Gumbel MLE per pair; AIC; tail dependence.
4. Stress analysis: calm vs stress (COVID, 2024 cocoa shock).
5. Networks: calm vs 2024-stress lower-tail dependence networks.
6. Rolling 250-day co-crash dynamics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk import copulas, evt, inference, marginals, networks  # noqa: E402

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"

STRESS_WINDOWS = {
    "covid": ("2020-03-01", "2020-06-30"),
    "cocoa_2024": ("2024-01-01", "2024-12-31"),
}

# Ingestion-script column -> canonical series name.
INGEST_MAP = {
    "cocoa_futures": "cocoa",
    "gold_futures": "gold",
    "gold_spot_proxy": "gold_spot",          # optional extra
    "brent_spot_fred": "brent",
    "wti_spot_fred": "wti",                  # spot preferred (no roll jumps)
    "wti_futures": "wti_fut",                # optional robustness extra
    "ghana_usd_exchange_daily": "ghs_usd",
}
CORE = ["cocoa", "gold", "brent", "wti", "ghs_usd"]  # pre-orientation names      # required
OPTIONAL_MIN_COVERAGE = 0.70                              # for extras

LABELS = {"cocoa": "Cocoa", "gold": "Gold", "gold_spot": "Gold spot",
          "brent": "Brent", "wti": "WTI", "wti_fut": "WTI fut",
          "ghs_usd": "GHS/USD", "cedi": "Cedi"}
NODE_COLORS = {"Cocoa": "#8B4513", "Gold": "#B8860B", "Gold spot": "#DAA520",
               "Brent": "#2F4F4F", "WTI": "#556B2F", "WTI fut": "#6B8E23",
               "GHS/USD": "#8B0000", "Cedi": "#8B0000"}


def label(c: str) -> str:
    return LABELS.get(c, c.replace("_", " ").title())


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def find_ingest_prices(ingest_dir: str | None) -> Path:
    candidates = []
    if ingest_dir:
        candidates.append(Path(ingest_dir) / "data_processed" / "daily_prices_wide.csv")
        candidates.append(Path(ingest_dir) / "daily_prices_wide.csv")
    for base in (ROOT, ROOT.parent, Path.cwd()):
        candidates.append(base / "project2_tail_dependence_data" / "data_processed"
                          / "daily_prices_wide.csv")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "daily_prices_wide.csv not found. Run the ingestion script first, or "
        "pass --ingest-dir /path/to/project2_tail_dependence_data")


def load_real_data(ingest_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt the ingestion script's wide price panel to the pipeline schema.

    Cleaning decisions (documented, deliberate):
    * Column mapping per INGEST_MAP; unknown columns are ignored with a note.
    * WTI/Brent use the FRED *spot* series as primary (futures kept as an
      optional extra) — spot has no roll jumps.
    * Optional extras (gold_spot, wti_fut) are kept only if they cover at
      least 70% of the aligned sample; Yahoo's XAUUSD=X is often spotty.
    * Calendar: business-day index, forward-fill gaps up to 3 days
      (holidays), never longer — stale prices must not fabricate zero-vol
      stretches. Rows still missing any CORE series are dropped.
    * GHS/USD: |daily log return| > 15% is masked as a bad tick (Yahoo's
      GHS series is known to contain stale quotes and spikes); masked days
      are logged, then forward-filled within the 3-day limit.
    * Returns are recomputed here as 100 * log-diff on the cleaned panel —
      the ingestion file's per-series returns are not reused because they
      were computed before cross-series calendar alignment.
    """
    path = find_ingest_prices(ingest_dir)
    print(f"Loading ingestion output: {path}")
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()

    known = {c: INGEST_MAP[c] for c in raw.columns if c in INGEST_MAP}
    ignored = [c for c in raw.columns if c not in INGEST_MAP]
    if ignored:
        print(f"  Ignoring unmapped columns: {ignored}")
    px = raw[list(known)].rename(columns=known)
    px = px.apply(pd.to_numeric, errors="coerce")

    missing = [c for c in CORE if c not in px.columns]
    if missing:
        raise ValueError(f"Ingestion file lacks required series: {missing}")

    # GHS bad-tick mask before alignment
    lr = np.log(px["ghs_usd"].where(px["ghs_usd"] > 0)).diff()
    bad = lr.abs() > 0.15
    if bad.any():
        print(f"  Masked {int(bad.sum())} suspect GHS/USD ticks (>15%/day):"
              f" {list(px.index[bad].date)[:8]}{' …' if bad.sum() > 8 else ''}")
        px.loc[bad, "ghs_usd"] = np.nan

    # Business-day alignment; limited forward fill
    px = px.asfreq("B").ffill(limit=3)

    # Drop optional extras with poor coverage; then require complete CORE rows
    for extra in [c for c in px.columns if c not in CORE]:
        cov = px[extra].notna().mean()
        if cov < OPTIONAL_MIN_COVERAGE:
            print(f"  Dropping optional series '{extra}' (coverage {cov:.0%})")
            px = px.drop(columns=extra)
    px = px.dropna(subset=CORE)
    # any extra still holey after CORE filter: fill tiny gaps then drop rows
    px = px.dropna()

    nonpos = (px <= 0).sum()
    if nonpos.any():
        print("  Non-positive prices set NaN (e.g. 2020-04-20 WTI):",
              nonpos[nonpos > 0].to_dict())
        px = px.where(px > 0).dropna()

    rets = 100 * np.log(px).diff().dropna()

    # ORIENTATION: ghs_usd (GHS per USD) has depreciation = positive return.
    # Flip its return sign and rename to 'cedi' so the LOWER tail means
    # bad-for-Ghana for every series (commodity crash / cedi depreciation).
    if "ghs_usd" in rets:
        rets["ghs_usd"] = -rets["ghs_usd"]
        rets = rets.rename(columns={"ghs_usd": "cedi"})
        px = px.rename(columns={"ghs_usd": "cedi"})
        print("  Oriented cedi: returns sign-flipped (lower tail = depreciation).")
    px = px.loc[rets.index.min():]

    px.to_csv(PROC / "prices_real.csv")
    rets.to_csv(PROC / "returns_real.csv")
    print(f"  Canonical panel written: {rets.shape[0]} obs x {rets.shape[1]} series "
          f"({', '.join(rets.columns)}), {rets.index.min().date()} → "
          f"{rets.index.max().date()}")
    return px, rets


def load_data(data: str, ingest_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data == "real":
        # Canonical BoG-based files by default. The alternate ingestion
        # path is only attempted when the caller explicitly passes
        # --ingest-dir; previously this was tried unconditionally first,
        # meaning the validated canonical panel could be silently
        # replaced by an unrelated ingestion pipeline's output if that
        # file ever happened to exist, with no explicit signal that this
        # had happened.
        if ingest_dir is not None:
            return load_real_data(ingest_dir)
        prices = pd.read_csv(PROC / "prices_real.csv", index_col=0, parse_dates=True)
        rets = pd.read_csv(PROC / "returns_real.csv", index_col=0, parse_dates=True)
        return prices, rets
    prices = pd.read_csv(PROC / f"prices_{data}.csv", index_col=0, parse_dates=True)
    rets = pd.read_csv(PROC / f"returns_{data}.csv", index_col=0, parse_dates=True)
    return prices, rets


def stress_flag(idx: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(False, index=idx)
    for a, b in STRESS_WINDOWS.values():
        flag |= (idx >= a) & (idx <= b)
    return flag


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main(data: str = "synthetic", ingest_dir: str | None = None) -> None:
    prices, rets = load_data(data, ingest_dir)
    print(f"Loaded {data} returns: {rets.shape[0]} obs x {rets.shape[1]} series")

    # ---------------------------------------------------------- 1. marginals
    fits, spec_log = marginals.fit_all(rets, gate=True)
    pd.DataFrame(spec_log).to_csv(TAB / "marginal_spec_search.csv", index=False)
    pit = marginals.pit_frame(fits)
    resid = marginals.std_resid_frame(fits)
    marg_rows = []
    for name, f in fits.items():
        p = f.params
        marg_rows.append({
            "series": label(name), "spec": f.spec, "adequate": f.adequate,
            "nu": f.nu, "AIC": f.aic,
            **inference.marginal_diagnostics(f.std_resid, f.pit),
        })
        if not f.adequate:
            print(f"  !! MARGINAL UNRESOLVED for {label(name)}: no ladder "
                  f"specification passed the adequacy gate. The deterministic "
                  f"reference fit ({f.spec}) is retained for pipeline "
                  f"continuity only, not as a preferred model. This is NOT "
                  f"inert for downstream inference: cross-environment "
                  f"replication showed the resulting rank ordering, "
                  f"including extreme-tail membership, is itself unstable "
                  f"(see diagnostics/diagnose_cedi_v3.py and DECISIONS.md). "
                  f"Results involving this series must be interpreted "
                  f"through the alternative-treatment robustness analysis, "
                  f"not from this single fit.")
    pd.DataFrame(marg_rows).round(3).to_csv(TAB / "marginal_garch.csv", index=False)
    pit.to_csv(TAB / f"_pit_{data}.csv")      # persisted for stages 3-6
    resid.to_csv(TAB / f"_resid_{data}.csv")

    vol_series = [c for c in ["cocoa", "gold", "brent"] if c in fits] or list(fits)[:3]
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in vol_series:
        ax.plot(fits[c].cond_vol, lw=0.8, label=label(c))
    for a, b in STRESS_WINDOWS.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.08)
    ax.set_ylabel("Conditional volatility (% / day)")
    ax.legend(frameon=False)
    ax.set_title("GJR-GARCH conditional volatility (stress windows shaded)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_conditional_vol.png", dpi=160)
    plt.close(fig)

    # --------------------------------------------------------------- 2. EVT
    evt_rows = []
    for c in rets.columns:
        losses = -resid[c]
        pot = evt.fit_pot(losses, quantile=0.90)
        k = max(25, int(0.05 * losses.size))
        evt_rows.append({
            "series": label(c), "u(q90)": pot.threshold, "xi": pot.xi,
            "beta": pot.beta, "n_exceed": pot.n_exceed,
            "VaR99_resid": pot.var(0.99), "ES99_resid": pot.es(0.99),
            "hill_gamma(k=5%)": evt.hill_estimator(losses, k),
        })
    pd.DataFrame(evt_rows).round(3).to_csv(TAB / "evt_pot_gpd.csv", index=False)

    sens_target = "cocoa" if "cocoa" in resid else resid.columns[0]
    sens = evt.threshold_sensitivity(-resid[sens_target])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sens["quantile"], sens["xi"], marker="o", ms=3, color="#8B4513")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel(r"GPD shape $\hat\xi$")
    ax.set_title(f"{label(sens_target)} loss tail: GPD shape vs threshold "
                 "(stability check)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_threshold_sensitivity.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ 3. copulas
    pairs = list(itertools.combinations(pit.columns, 2))
    cop_rows, lam_rows = [], []
    for a, b in pairs:
        u, v = pit[a].to_numpy(), pit[b].to_numpy()
        fitd = copulas.fit_pair(u, v)
        best = min(fitd.values(), key=lambda f: f.aic)
        for fam, f in fitd.items():
            cop_rows.append({
                "pair": f"{label(a)}–{label(b)}", "family": fam,
                "loglik": f.loglik, "AIC": f.aic, **f.params,
                "best": fam == best.family,
            })
        lam_rows.append({
            "pair": f"{label(a)}–{label(b)}",
            "lambda_L_t": fitd["t"].lambda_lower,
            "lambda_L_clayton": fitd["clayton"].lambda_lower,
            "lambda_U_gumbel": fitd["gumbel"].lambda_upper,
            "lambda_L_empirical(q=5%)": copulas.empirical_lambda_lower(u, v, 0.05),
            "lambda_U_empirical(q=95%)": copulas.empirical_lambda_upper(u, v, 0.95),
            "best_family": best.family,
        })
    pd.DataFrame(cop_rows).round(4).to_csv(TAB / "copula_fits.csv", index=False)
    lam_df = pd.DataFrame(lam_rows).round(3)
    lam_df.to_csv(TAB / "tail_dependence.csv", index=False)

    # ----------------------------------------------- 4. calm vs stress split
    flag = stress_flag(pit.index)
    pit_calm, pit_stress = pit[~flag], pit[flag]
    # Small stress windows make lambda(q=0.05) very noisy; use q=0.10 for
    # the inferential comparison and report block-bootstrap 95% CIs plus a
    # CI for the stress-calm difference (significance = CI excludes 0).
    q_inf = 0.10
    rows = []
    for a, b in pairs:
        res = inference.calm_stress_difference(
            copulas.pseudo_obs(pit_calm[[a]])[:, 0],
            copulas.pseudo_obs(pit_calm[[b]])[:, 0],
            copulas.pseudo_obs(pit_stress[[a]])[:, 0],
            copulas.pseudo_obs(pit_stress[[b]])[:, 0],
            q=q_inf, n_boot=500)
        rows.append({"pair": f"{label(a)}–{label(b)}",
                     "lambda_L_calm": res["lambda_calm"],
                     "calm_95CI": f"[{res['calm_lo']:.2f}, {res['calm_hi']:.2f}]",
                     "lambda_L_stress": res["lambda_stress"],
                     "stress_95CI": f"[{res['stress_lo']:.2f}, {res['stress_hi']:.2f}]",
                     "difference": res["difference"],
                     "diff_95CI": f"[{res['diff_lo']:.2f}, {res['diff_hi']:.2f}]",
                     "significant_5pct": res["significant_5pct"]})
    split = pd.DataFrame(rows)
    split["ratio"] = (split["lambda_L_stress"]
                      / split["lambda_L_calm"].replace(0, np.nan))
    split.round(3).to_csv(TAB / "calm_vs_stress_tail_dependence.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(split)), 4.5))
    x = np.arange(len(split))
    ax.bar(x - 0.2, split["lambda_L_calm"], width=0.38, label="Calm",
           color="#4477AA")
    ax.bar(x + 0.2, split["lambda_L_stress"], width=0.38, label="Stress",
           color="#CC3311")
    ax.set_xticks(x, split["pair"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"Empirical $\hat\lambda_L(q=0.10)$ with 95% CIs in table")
    ax.set_title("Lower-tail dependence: calm vs stress periods")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_calm_vs_stress.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ 5. network
    def sub_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        po = pd.DataFrame(copulas.pseudo_obs(frame), columns=frame.columns)
        m = networks.tail_dependence_matrix(po, q=0.05)
        m.index = m.columns = [label(c) for c in m.columns]
        return m

    lam_calm = sub_matrix(pit_calm)
    c24 = (pit.index >= STRESS_WINDOWS["cocoa_2024"][0]) & (
        pit.index <= STRESS_WINDOWS["cocoa_2024"][1])
    if c24.sum() > 60:
        lam_2024 = sub_matrix(pit[c24])
    else:
        print("  <60 obs in 2024 window; using all stress days for the network.")
        lam_2024 = sub_matrix(pit_stress)
    lam_calm.round(3).to_csv(TAB / "lambda_matrix_calm.csv")
    lam_2024.round(3).to_csv(TAB / "lambda_matrix_2024.csv")
    networks.plot_networks(
        lam_calm, lam_2024,
        ("Calm periods", "2024 cocoa-shock window"),
        str(FIG / "fig5_tail_networks.png"),
        node_colors=NODE_COLORS,
    )

    # ---------------------------------------------------- 6. rolling dynamics
    roll = networks.rolling_lower_tail(pit, window=250, q=0.10)
    wanted = [("cocoa", "gold"), ("cocoa", "brent"), ("brent", "wti"),
              ("cocoa", "cedi"), ("brent", "cedi"), ("gold", "cedi"),
              ("cocoa", "ghs_usd"), ("brent", "ghs_usd")]
    key_pairs = [p for p in wanted
                 if p in roll or (p[1], p[0]) in roll][:6] or list(roll)[:6]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for a, b in key_pairs:
        s = roll[(a, b)] if (a, b) in roll else roll[(b, a)]
        ax.plot(s, lw=1.0, label=f"{label(a)}–{label(b)}")
    for a, b in STRESS_WINDOWS.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.08)
    ax.set_ylabel(r"Rolling $\hat\lambda_L(q=0.10)$, 250-day window")
    ax.set_title("Rolling lower-tail dependence (stress windows shaded)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_rolling_tail_dependence.png", dpi=160)
    plt.close(fig)
    roll_df = pd.concat(roll.values(), axis=1)
    roll_df.columns = [f"{label(a)}–{label(b)}" for a, b in roll]
    roll_df.to_csv(TAB / "rolling_lambda_L.csv")

    # prices overview
    fig, ax = plt.subplots(figsize=(11, 4))
    (prices / prices.iloc[0] * 100).plot(ax=ax, lw=0.9)
    ax.set_ylabel("Index (start = 100)")
    ax.set_title(f"Price levels — {data} data")
    ax.legend([label(c) for c in prices.columns], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig0_prices.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    summary = {
        "data": data,
        "series": [str(c) for c in rets.columns],
        "n_obs": int(pit.shape[0]),
        "period": [str(pit.index.min().date()), str(pit.index.max().date())],
        "evt_xi": {r["series"]: r["xi"] for r in evt_rows},
        "best_copula_by_pair": dict(zip(lam_df["pair"], lam_df["best_family"])),
        "mean_lambda_L_calm": float(split["lambda_L_calm"].mean()),
        "mean_lambda_L_stress": float(split["lambda_L_stress"].mean()),
        "n_pairs_significant_increase": int(
            ((split["difference"] > 0) & split["significant_5pct"]).sum()),
        "n_pairs_total": int(len(split)),
    }
    (TAB / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Done. Figures in outputs/figures, tables in outputs/tables.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--ingest-dir", default=None,
                    help="Path to project2_tail_dependence_data produced by the "
                         "ingestion script (auto-detected if omitted)")
    a = ap.parse_args()
    main(a.data, a.ingest_dir)
