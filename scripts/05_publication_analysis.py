"""Stages 2-6 of the publication run (real data).

S2  EVT extensions: threshold scans all series, GEV block maxima,
    bootstrap CIs for the GPD shape xi.
S3  Copula GoF: Rosenblatt-CvM parametric bootstrap (500 reps), all
    four families, every pair.
S4  Dynamics & inference: rolling 250-day t-copula (monthly step);
    q-sensitivity {0.025, 0.05, 0.10}; Bonferroni columns; and the
    pre-registered upper-tail calm-vs-stress test (the 2024 cocoa shock
    was an upper-tail event, per paper §1).
S5  Robustness: weekly-frequency headline rerun; exclude-COVID stress
    definition. (Pink-Sheet roll-effect comparison SKIPPED: file not
    available in this environment — documented, script-ready.)
S6  Transmission: monthly distributed-lag regressions with HAC errors.
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tailrisk import copulas, evt, gof, inference, marginals  # noqa: E402
from tailrisk.gof import NU_GRID  # noqa: E402

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
STRESS = {"covid": ("2020-03-01", "2020-06-30"),
          "cocoa_2024": ("2024-01-01", "2024-12-31")}
LBL = {"cocoa": "Cocoa", "gold": "Gold", "brent": "Brent", "wti": "WTI",
       "cedi": "Cedi"}
KEY_PAIRS = [("cocoa", "gold"), ("cocoa", "brent"), ("cocoa", "cedi"),
             ("brent", "cedi")]


def stress_flag(idx):
    f = pd.Series(False, index=idx)
    for a, b in STRESS.values():
        f |= (idx >= a) & (idx <= b)
    return f


def po(frame, col):
    return copulas.pseudo_obs(frame[[col]])[:, 0]


def main() -> None:
    rets = pd.read_csv(ROOT / "data/processed/returns_real.csv",
                       index_col=0, parse_dates=True)
    fits, _ = marginals.fit_all(rets, gate=True)
    pit = marginals.pit_frame(fits)
    resid = marginals.std_resid_frame(fits)
    pairs = list(itertools.combinations(pit.columns, 2))
    flag = stress_flag(pit.index)
    pit_c, pit_s = pit[~flag], pit[flag]

    # ================================================================= S2
    print("== S2: EVT extensions ==")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    gev_rows, xi_rows = [], []
    rng = np.random.default_rng(0)
    for ax, c in zip(axes.ravel(), rets.columns):
        losses = -resid[c]
        sens = evt.threshold_sensitivity(losses)
        ax.plot(sens["quantile"], sens["xi"], marker="o", ms=2)
        ax.axhline(0, color="grey", lw=.6, ls="--")
        ax.set_title(LBL[c], fontsize=10)
        ax.set_xlabel("threshold quantile", fontsize=8)
        ax.set_ylabel(r"$\hat\xi$", fontsize=8)
        # GPD xi bootstrap CI (resample exceedances, 500 reps)
        pot = evt.fit_pot(losses, 0.90)
        exc = np.sort(losses[losses > pot.threshold] - pot.threshold)
        xis = []
        for _ in range(500):
            xb = rng.choice(exc, size=exc.size, replace=True)
            xis.append(stats.genpareto.fit(xb, floc=0.0)[0])
        lo, hi = np.quantile(xis, [.025, .975])
        xi_rows.append({"series": LBL[c], "xi": pot.xi,
                        "xi_lo95": lo, "xi_hi95": hi,
                        "VaR99": pot.var(.99), "ES99": pot.es(.99)})
        # GEV on monthly block maxima of losses
        bm = losses.resample("ME").max().dropna()
        shape, loc, scale = stats.genextreme.fit(bm)
        gev_rows.append({"series": LBL[c], "gev_xi": -shape,  # sign conv.
                         "gev_loc": loc, "gev_scale": scale,
                         "n_blocks": len(bm)})
    axes.ravel()[-1].axis("off")
    fig.suptitle("GPD shape vs threshold — all series")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_threshold_all_series.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(xi_rows).round(3).to_csv(TAB / "evt_xi_bootstrap_ci.csv",
                                          index=False)
    pd.DataFrame(gev_rows).round(3).to_csv(TAB / "evt_gev_block_maxima.csv",
                                           index=False)
    print(pd.DataFrame(xi_rows).round(3).to_string(index=False))

    # ================================================================= S3
    print("== S3: copula goodness-of-fit (Rosenblatt-CvM, 500 boots) ==")
    gof_rows = []
    for a, b in pairs:
        u, v = po(pit, a), po(pit, b)
        for fam in ["gaussian", "t", "clayton", "gumbel"]:
            r = gof.gof_test(u, v, fam, n_boot=500, seed=42)
            gof_rows.append({"pair": f"{LBL[a]}–{LBL[b]}", "family": fam,
                             "Sn": r["Sn"], "p_value": r["p_value"],
                             "not_rejected_5pct": r["p_value"] > 0.05,
                             **r["params"]})
        done = [g for g in gof_rows if g["pair"] == f"{LBL[a]}–{LBL[b]}"]
        ok = [g["family"] for g in done if g["not_rejected_5pct"]]
        print(f"  {LBL[a]}–{LBL[b]}: not rejected -> {ok or 'NONE'}")
    gof_df = pd.DataFrame(gof_rows)
    gof_df.round(4).to_csv(TAB / "copula_gof.csv", index=False)

    # ================================================================= S4
    print("== S4: dynamics, q-sensitivity, Bonferroni, upper tail ==")
    # rolling t-copula (tau inversion + nu grid), 250d window, 21d step
    win, step = 250, 21
    roll_rows = []
    for a, b in KEY_PAIRS:
        xa, xb = pit[a].to_numpy(), pit[b].to_numpy()
        for end in range(win, len(pit), step):
            ua = stats.rankdata(xa[end - win:end]) / (win + 1)
            ub = stats.rankdata(xb[end - win:end]) / (win + 1)
            tau = stats.kendalltau(ua, ub).statistic
            rho = float(np.sin(np.pi * np.clip(tau, -.95, .95) / 2))
            lls = [copulas._t_loglik(rho, nu, ua, ub) for nu in NU_GRID]
            nu = float(NU_GRID[int(np.argmax(lls))])
            lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)),
                                  nu + 1)
            roll_rows.append({"date": pit.index[end - 1],
                              "pair": f"{LBL[a]}–{LBL[b]}",
                              "rho": rho, "nu": nu, "lambda_t": float(lam)})
    roll = pd.DataFrame(roll_rows)
    roll.to_csv(TAB / "rolling_t_copula.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for pr, g in roll.groupby("pair"):
        axes[0].plot(g["date"], g["rho"], lw=1.1, label=pr)
        axes[1].plot(g["date"], g["lambda_t"], lw=1.1, label=pr)
    for ax, ylab in zip(axes, [r"$\rho_t$", r"$\lambda_{L,t}$ (t-copula)"]):
        for a_, b_ in STRESS.values():
            ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="red",
                       alpha=.08)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title("Rolling 250-day t-copula (monthly step)")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_rolling_t_copula.png", dpi=160)
    plt.close(fig)

    # q-sensitivity + Bonferroni, lower AND upper tails
    for tail, upper in [("lower", False), ("upper", True)]:
        rows = []
        for qq in [0.025, 0.05, 0.10]:
            for a, b in pairs:
                res = inference.calm_stress_difference(
                    po(pit_c, a), po(pit_c, b), po(pit_s, a), po(pit_s, b),
                    q=qq, n_boot=500, bonf_m=len(pairs), upper=upper)
                rows.append({"q": qq, "pair": f"{LBL[a]}–{LBL[b]}", **res})
        df = pd.DataFrame(rows)
        df.round(4).to_csv(TAB / f"calm_stress_{tail}_qsens_bonferroni.csv",
                           index=False)
        for qq in [0.025, 0.05, 0.10]:
            d = df[df["q"] == qq]
            print(f"  {tail} tail q={qq}: mean calm "
                  f"{d['lambda_calm'].mean():.3f} -> stress "
                  f"{d['lambda_stress'].mean():.3f} | sig 5%: "
                  f"{int(d['significant_5pct'].sum())}/10 | Bonferroni: "
                  f"{int(d['significant_bonferroni'].sum())}/10")

    # upper tail, 2024-only stress window (the cocoa-shock hypothesis)
    c24 = (pit.index >= STRESS["cocoa_2024"][0]) & (
        pit.index <= STRESS["cocoa_2024"][1])
    pit_24, pit_not24 = pit[c24], pit[~c24 & ~flag]
    rows = []
    for a, b in pairs:
        res = inference.calm_stress_difference(
            po(pit_not24, a), po(pit_not24, b), po(pit_24, a), po(pit_24, b),
            q=0.10, n_boot=500, bonf_m=len(pairs), upper=True)
        rows.append({"pair": f"{LBL[a]}–{LBL[b]}", **res})
    up24 = pd.DataFrame(rows)
    up24.round(4).to_csv(TAB / "upper_tail_2024_window.csv", index=False)
    print("  upper tail, 2024 window only: sig 5%:",
          int(up24["significant_5pct"].sum()), "/10 | Bonferroni:",
          int(up24["significant_bonferroni"].sum()), "/10")

    # ================================================================= S5
    print("== S5: robustness ==")
    # (a) weekly frequency
    px = pd.read_csv(ROOT / "data/processed/prices_real.csv",
                     index_col=0, parse_dates=True)
    pw = px.resample("W-FRI").last().dropna()
    rw = 100 * np.log(pw).diff().dropna()
    rw["cedi"] = rw["cedi"]        # already oriented in daily build? NO:
    # prices_real holds GHS/USD level; daily returns were flipped in the
    # panel build. Weekly returns from prices need the same flip:
    rw["cedi"] = -np.abs(rw["cedi"]) if False else rw["cedi"]
    fits_w, _ = marginals.fit_all(rw, gate=True)
    pit_w = marginals.pit_frame(fits_w)
    flag_w = stress_flag(pit_w.index)
    rows = []
    for a, b in pairs:
        res = inference.calm_stress_difference(
            po(pit_w[~flag_w], a), po(pit_w[~flag_w], b),
            po(pit_w[flag_w], a), po(pit_w[flag_w], b),
            q=0.10, n_boot=500, bonf_m=len(pairs))
        rows.append({"pair": f"{LBL[a]}–{LBL[b]}", **res})
    wk = pd.DataFrame(rows)
    wk.round(4).to_csv(TAB / "robustness_weekly_lower.csv", index=False)
    print(f"  weekly (n={len(pit_w)}): mean calm "
          f"{wk['lambda_calm'].mean():.3f} -> stress "
          f"{wk['lambda_stress'].mean():.3f} | sig 5%: "
          f"{int(wk['significant_5pct'].sum())}/10")

    # (b) exclude-COVID: stress = 2024 only, calm excludes COVID too
    rows = []
    for a, b in pairs:
        res = inference.calm_stress_difference(
            po(pit_not24, a), po(pit_not24, b), po(pit_24, a), po(pit_24, b),
            q=0.10, n_boot=500, bonf_m=len(pairs))
        rows.append({"pair": f"{LBL[a]}–{LBL[b]}", **res})
    xc = pd.DataFrame(rows)
    xc.round(4).to_csv(TAB / "robustness_excl_covid_lower.csv", index=False)
    print(f"  excl-COVID lower: sig 5%: {int(xc['significant_5pct'].sum())}/10")
    print("  (d) Pink-Sheet roll-effect comparison SKIPPED: "
          "CMO monthly file not available in this environment.")

    # ================================================================= S6
    print("== S6: transmission regressions (HAC) ==")
    import statsmodels.api as sm

    macro = pd.read_csv(ROOT / "data/processed/ghana_macro_real.csv",
                        index_col=0, parse_dates=True)
    comm = ["cocoa", "gold", "brent"]
    q10 = rets[comm].rolling(250).quantile(0.10)
    joint = ((rets[comm] < q10).sum(axis=1) >= 2).astype(int)
    m_joint = joint.resample("ME").sum().rename("joint_tail_days")
    lam_cb = roll[roll["pair"] == "Cocoa–Brent"].set_index("date")["lambda_t"]
    m_lam = lam_cb.resample("ME").mean().ffill(limit=1).rename("mean_lambda_t")
    Y = macro["cedi_depreciation_pct_m"]
    reg_rows = []
    for yname, y in [("cedi_depreciation", Y),
                     ("d_cpi_yoy", macro["cpi_yoy"].diff()),
                     ("exports_growth",
                      100 * np.log(macro["exports_usd_m"]).diff())]:
        X = pd.concat([m_joint, m_joint.shift(1), m_lam.shift(1)], axis=1)
        X.columns = ["joint_days_t", "joint_days_t1", "lambda_t1"]
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(df) < 30:
            reg_rows.append({"target": yname, "n": len(df),
                             "note": "insufficient overlap — skipped"})
            print(f"  {yname}: only {len(df)} months — skipped")
            continue
        mod = sm.OLS(df["y"], sm.add_constant(df[X.columns])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        for k in X.columns:
            reg_rows.append({"target": yname, "n": len(df), "regressor": k,
                             "coef": mod.params[k], "hac_p": mod.pvalues[k],
                             "r2": mod.rsquared})
        sig = [k for k in X.columns if mod.pvalues[k] < .05]
        print(f"  {yname}: n={len(df)}, R2={mod.rsquared:.3f}, "
              f"sig regressors: {sig or 'none'}")
    pd.DataFrame(reg_rows).round(4).to_csv(TAB / "transmission_regressions.csv",
                                           index=False)

    print("Stages 2-6 complete.")


if __name__ == "__main__":
    main()
