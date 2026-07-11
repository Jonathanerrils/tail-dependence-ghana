"""End-to-end tail-dependence pipeline.

Usage:
    python scripts/02_run_pipeline.py            # synthetic demo data
    python scripts/02_run_pipeline.py --data real

Stages
------
1. Marginals: AR(1)-GJR-GARCH(1,1)-t per series; PIT residuals.
2. EVT: POT/GPD on loss tails of standardized residuals; threshold
   sensitivity for cocoa; Hill estimates.
3. Copulas: Gaussian/t/Clayton/Gumbel MLE per pair; AIC comparison;
   analytic vs empirical tail dependence.
4. Stress analysis: calm vs stress subsamples (COVID, 2024 cocoa shock).
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

from tailrisk import copulas, evt, marginals, networks  # noqa: E402

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
STRESS_WINDOWS = {
    "covid": ("2020-03-01", "2020-06-30"),
    "cocoa_2024": ("2024-01-01", "2024-12-31"),
}
LABELS = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI",
          "ghs_usd": "GHS/USD"}
NODE_COLORS = {"Cocoa": "#8B4513", "Gold": "#B8860B", "Brent": "#2F4F4F",
               "WTI": "#556B2F", "GHS/USD": "#8B0000"}


def stress_flag(idx: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(False, index=idx)
    for a, b in STRESS_WINDOWS.values():
        flag |= (idx >= a) & (idx <= b)
    return flag


def main(data: str = "synthetic") -> None:
    rets = pd.read_csv(ROOT / "data" / "processed" / f"returns_{data}.csv",
                       index_col=0, parse_dates=True)
    prices = pd.read_csv(ROOT / "data" / "processed" / f"prices_{data}.csv",
                         index_col=0, parse_dates=True)
    print(f"Loaded {data} returns: {rets.shape[0]} obs x {rets.shape[1]} series")

    # ---------------------------------------------------------- 1. marginals
    fits = marginals.fit_all(rets)
    pit = marginals.pit_frame(fits)
    resid = marginals.std_resid_frame(fits)
    marg_rows = []
    for name, f in fits.items():
        p = f.params
        marg_rows.append({
            "series": LABELS[name], "mu": p.get("Const", np.nan),
            "alpha": p.get("alpha[1]", np.nan), "gamma": p.get("gamma[1]", np.nan),
            "beta": p.get("beta[1]", np.nan), "nu": f.nu, "AIC": f.aic,
        })
    pd.DataFrame(marg_rows).round(3).to_csv(TAB / "marginal_garch.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    for c in ["cocoa", "gold", "brent"]:
        ax.plot(fits[c].cond_vol, lw=0.8, label=LABELS[c])
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
            "series": LABELS[c], "u(q90)": pot.threshold, "xi": pot.xi,
            "beta": pot.beta, "n_exceed": pot.n_exceed,
            "VaR99_resid": pot.var(0.99), "ES99_resid": pot.es(0.99),
            "hill_gamma(k=5%)": evt.hill_estimator(losses, k),
        })
    evt_df = pd.DataFrame(evt_rows).round(3)
    evt_df.to_csv(TAB / "evt_pot_gpd.csv", index=False)

    sens = evt.threshold_sensitivity(-resid["cocoa"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sens["quantile"], sens["xi"], marker="o", ms=3, color="#8B4513")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel(r"GPD shape $\hat\xi$")
    ax.set_title("Cocoa loss tail: GPD shape vs threshold (stability check)")
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
                "pair": f"{LABELS[a]}–{LABELS[b]}", "family": fam,
                "loglik": f.loglik, "AIC": f.aic, **f.params,
                "best": fam == best.family,
            })
        lam_rows.append({
            "pair": f"{LABELS[a]}–{LABELS[b]}",
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
    rows = []
    for a, b in pairs:
        rows.append({
            "pair": f"{LABELS[a]}–{LABELS[b]}",
            "lambda_L_calm": copulas.empirical_lambda_lower(
                copulas.pseudo_obs(pit_calm[[a]])[:, 0],
                copulas.pseudo_obs(pit_calm[[b]])[:, 0], 0.05),
            "lambda_L_stress": copulas.empirical_lambda_lower(
                copulas.pseudo_obs(pit_stress[[a]])[:, 0],
                copulas.pseudo_obs(pit_stress[[b]])[:, 0], 0.05),
        })
    split = pd.DataFrame(rows)
    split["ratio"] = (split["lambda_L_stress"] / split["lambda_L_calm"].replace(0, np.nan))
    split.round(3).to_csv(TAB / "calm_vs_stress_tail_dependence.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(split))
    ax.bar(x - 0.2, split["lambda_L_calm"], width=0.38, label="Calm", color="#4477AA")
    ax.bar(x + 0.2, split["lambda_L_stress"], width=0.38, label="Stress",
           color="#CC3311")
    ax.set_xticks(x, split["pair"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"Empirical $\hat\lambda_L(q=0.05)$")
    ax.set_title("Lower-tail dependence: calm vs stress periods")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_calm_vs_stress.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ 5. network
    def sub_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        po = pd.DataFrame(copulas.pseudo_obs(frame), columns=frame.columns)
        m = networks.tail_dependence_matrix(po, q=0.05)
        m.index = m.columns = [LABELS[c] for c in m.columns]
        return m

    lam_calm = sub_matrix(pit_calm)
    c24 = (pit.index >= STRESS_WINDOWS["cocoa_2024"][0]) & (
        pit.index <= STRESS_WINDOWS["cocoa_2024"][1])
    lam_2024 = sub_matrix(pit[c24])
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
    key_pairs = [("cocoa", "gold"), ("cocoa", "brent"), ("brent", "wti"),
                 ("cocoa", "ghs_usd"), ("brent", "ghs_usd")]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for a, b in key_pairs:
        s = roll[(a, b)] if (a, b) in roll else roll[(b, a)]
        ax.plot(s, lw=1.0, label=f"{LABELS[a]}–{LABELS[b]}")
    for a, b in STRESS_WINDOWS.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.08)
    ax.set_ylabel(r"Rolling $\hat\lambda_L(q=0.10)$, 250-day window")
    ax.set_title("Rolling lower-tail dependence (stress windows shaded)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_rolling_tail_dependence.png", dpi=160)
    plt.close(fig)
    pd.concat(roll.values(), axis=1).to_csv(TAB / "rolling_lambda_L.csv")

    # prices overview figure
    fig, ax = plt.subplots(figsize=(11, 4))
    (prices / prices.iloc[0] * 100).plot(ax=ax, lw=0.9)
    ax.set_ylabel("Index (start = 100)")
    ax.set_title(f"Price levels — {data} data")
    ax.legend([LABELS[c] for c in prices.columns], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig0_prices.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    summary = {
        "data": data,
        "n_obs": int(pit.shape[0]),
        "period": [str(pit.index.min().date()), str(pit.index.max().date())],
        "evt_xi": {r["series"]: r["xi"] for r in evt_rows},
        "best_copula_by_pair": dict(zip(lam_df["pair"], lam_df["best_family"])),
        "mean_lambda_L_calm": float(split["lambda_L_calm"].mean()),
        "mean_lambda_L_stress": float(split["lambda_L_stress"].mean()),
    }
    (TAB / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Done. Figures in outputs/figures, tables in outputs/tables.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    main(ap.parse_args().data)
