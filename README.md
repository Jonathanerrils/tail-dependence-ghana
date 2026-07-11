# Tail Dependence and Extreme Commodity Risk in Ghana

**A copula-EVT analysis of cocoa, gold and crude oil, with transmission to the Ghana cedi and macro indicators.**

Do cocoa, gold and crude oil co-move more strongly during extreme market conditions than in normal periods — and does that joint tail risk transmit into Ghanaian exchange-rate, inflation and export-revenue risk? This repository contains a complete, reproducible pipeline answering that question: GARCH-filtered margins, peaks-over-threshold extreme value theory, four copula families with analytic and nonparametric tail-dependence estimation, calm-versus-stress tail-risk networks, rolling co-crash dynamics, an interactive dashboard, and a working-paper draft.

> ⚠️ **Data status.** The committed results are a *validation run on synthetic data* engineered to embed the stylized facts under study (volatility clustering, regime-dependent tail dependence, the 2024 cocoa shock, a cedi transmission channel). This proves the pipeline recovers known structure end-to-end. Run `scripts/01_fetch_real_data.py` locally, then rerun the pipeline with `--data real`, before treating any number as an empirical finding.

## Headline of the validation run

The Student-t copula beats the tail-independent Gaussian by AIC on **all ten pairs**; average empirical lower-tail dependence more than doubles from calm (λ̂_L ≈ 0.15) to stress periods (λ̂_L ≈ 0.34); and the 2024 cocoa-shock network is visibly denser than the calm network, with the commodity–cedi edges strengthening most.

| | |
|---|---|
| ![networks](outputs/figures/fig5_tail_networks.png) | ![rolling](outputs/figures/fig4_rolling_tail_dependence.png) |

## Repository structure

```
src/tailrisk/          Library code
  marginals.py         AR(1)-GJR-GARCH(1,1)-t filtering + PIT
  evt.py               POT/GPD, VaR/ES, Hill, threshold sensitivity, mean excess
  copulas.py           Gaussian / t / Clayton / Gumbel MLE + tail dependence
  networks.py          Rolling co-crash estimation + tail-risk network plots
scripts/
  00_generate_synthetic_data.py   Synthetic demo dataset (seeded, documented)
  01_fetch_real_data.py           Real data via yfinance + World Bank (run locally)
  02_run_pipeline.py              Full analysis → outputs/figures, outputs/tables
  03_build_dashboard.py           Interactive HTML dashboard
data/processed/        Returns/prices (synthetic committed; real generated locally)
outputs/figures        fig0–fig5 (prices, vol, EVT stability, calm-vs-stress,
                       rolling tail dependence, tail networks)
outputs/tables         GARCH, GPD, copula, tail-dependence and network matrices
outputs/dashboard.html Interactive dashboard (heatmaps, rolling λ_L, event timeline)
paper/paper.md         Working-paper draft (methodology complete)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce the committed validation run
python scripts/00_generate_synthetic_data.py
python scripts/02_run_pipeline.py
python scripts/03_build_dashboard.py

# Real data (requires internet)
pip install yfinance wbgapi
python scripts/01_fetch_real_data.py
python scripts/02_run_pipeline.py --data real
python scripts/03_build_dashboard.py --data real
```

## Method summary

1. **Marginals.** Each daily return series is filtered by AR(1)-GJR-GARCH(1,1) with Student-t innovations; standardized residuals are mapped to pseudo-uniforms (PIT, then rank-based pseudo-observations) so the copula layer sees approximately i.i.d. uniform margins (IFM / pseudo-MLE).
2. **EVT.** Loss tails of the standardized residuals are fitted with a generalized Pareto distribution above the 90% threshold; ξ̂ is scanned across the 85–97.5% thresholds as a stability check, with the Hill estimator as a semi-parametric cross-check.
3. **Copulas.** Gaussian, Student-t, Clayton and Gumbel copulas are estimated per pair by pseudo-MLE and compared by AIC; tail dependence is reported analytically (from t/Clayton/Gumbel parameters) and nonparametrically (λ̂_L(q) = P(U≤q, V≤q)/q).
4. **Stress & networks.** Ex-ante stress windows (COVID Mar–Jun 2020; calendar-2024 cocoa shock) versus calm subsamples; λ̂_L matrices rendered as weighted networks; 250-day rolling co-crash dynamics.
5. **Transmission.** Daily tail-risk aggregates (joint-exceedance counts, mean rolling λ̂_L) are related to monthly cedi, inflation and export-revenue movements (second-stage extension).

## Data sources (real run)

Futures and FX from Yahoo Finance (CC=F, GC=F, BZ=F, CL=F, GHS=X). Ghana macro: Bank of Ghana time-series workbooks (monthly CPI, cedi reference rate, export receipts) from bog.gov.gh → Statistics → Time Series Data; World Bank `FP.CPI.TOTL.ZG` as a cross-check. Save BoG workbooks under `data/raw/`.

## Publishing this repository to GitHub

```bash
cd tail-dependence-ghana
git remote add origin https://github.com/<your-username>/tail-dependence-ghana.git
git branch -M main
git push -u origin main
```

## Citation

> *Tail Dependence and Extreme Commodity Risk in Ghana: A Copula-EVT Analysis of Cocoa, Gold and Crude Oil.* Working paper, 2026. See `paper/paper.md`.

## License

MIT — see `LICENSE`.
