# Extreme Commodity Tail Risk and Exchange-Rate Exposure in Ghana

**A copula–EVT analysis of cocoa, gold and crude oil, with the Bank of
Ghana's official interbank rate as the exchange-rate reference series.**

Full manuscript: [`paper/latex/paper.tex`](paper/latex/paper.tex)
([compiled PDF](paper/latex/paper.pdf)), target journal *Risks*.
`paper/paper.md` is an earlier, superseded draft, retained for history
only — do not cite it.

## Headline findings

**Economic exposure.** Across ten commodity/cedi pairs, three quantile
levels, two tails, and three stress-window definitions (COVID-2020,
the 2024 cocoa supply shock, and their combination) — ninety
simultaneous hypothesis tests — **zero cells survive correction for
multiple testing**, under both a full family-wise Bonferroni correction
and an independent Benjamini–Hochberg FDR procedure. No robust evidence
that tail dependence among cocoa, gold and crude oil intensifies during
stress, and no detectable commodity-to-cedi transmission channel at
daily-to-monthly frequency.

**Model risk.** Expanding the candidate copula set from the
conventional four families (Gaussian, Student-$t$, Clayton, Gumbel) to
eight (adding Frank, Joe, and the survival/rotated Clayton and Gumbel)
does not rescue the six pairs — all four commodity-commodity pairs plus
Brent–WTI — whose dependence structure the narrower set already could
not fit. Every one of eight families is rejected by formal
goodness-of-fit testing for all six. This is evidence of a genuine
model-specification gap, not of absent dependence.

**The cedi's own risk.** Independently of any commodity link, the
cedi's extreme-value shape parameter ($\hat\xi = 0.67$) is roughly four
times the heaviest commodity tail in the panel, and resists six
independent attempts at adequate marginal modelling — including an
explicit zero-inflated hurdle model and a two-state Markov-switching
extension — though the paper's substantive tail-dependence conclusions
are stable across all six.

| | |
|---|---|
| ![networks](outputs/figures/fig5_tail_networks.png) | ![rolling](outputs/figures/fig4_rolling_tail_dependence.png) |

## Repository structure

```
paper/latex/            Full manuscript (paper.tex, paper.pdf, figures, bib)
paper/paper.md           Superseded early draft — do not cite
paper/VERIFICATION_LEDGER.csv   Every quantitative claim mapped to its source

src/tailrisk/            Library code
  marginals.py            Adequacy-gated AR-GJR-GARCH/EGARCH-t filtering + PIT
  evt.py                  POT/GPD, VaR/ES, Hill, threshold sensitivity
  copulas.py               8-family copula MLE (Gaussian/t/Clayton/Gumbel/
                           Frank/Joe/survival Clayton/survival Gumbel)
  gof.py                  Rosenblatt-CvM goodness-of-fit, parametric bootstrap
  inference.py             Moving-block bootstrap, Bonferroni + FDR correction
  networks.py              Rolling co-crash estimation + tail-risk networks

scripts/
  00_generate_synthetic_data.py    Synthetic validation dataset (seeded)
  01_fetch_real_data.py            Real data via yfinance (run locally)
  02_run_pipeline.py                Core pipeline: marginals/EVT/copulas/networks
  03_build_dashboard.py             Interactive HTML dashboard
  04_integrate_bog.py               Bank of Ghana cedi integration (authoritative)
  05_publication_analysis.py        EVT extensions, rolling t-copula, robustness
  06_gof_8family.py                 Full 8-family goodness-of-fit sweep
  07_full60_correction.py           Full 60-cell Bonferroni + FDR correction
  08_disaggregated_stress.py        COVID-only / 2024-only stress windows
  09_pinksheet_check.py             World Bank Pink Sheet roll-effect validation
  10_cedi_alternatives.py           Cedi marginal robustness (raw/ARMA/weekly)
  11_cedi_hurdle.py                 Cedi hurdle (zero-inflated) model
  12_cedi_mixture.py                Cedi two-regime Gaussian mixture
  13_cedi_markov.py                 Cedi 2-state Markov-switching model

outputs/tables/          All result tables (80+ files); see paper for index
outputs/figures/         All figures referenced in the paper
data/raw/bog/             Bank of Ghana source tables (redistributed, attributed)
data/processed/           Synthetic validation data (real data gitignored — see below)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Real data (requires internet; Yahoo Finance data is fetch-only, not redistributed)
python scripts/01_fetch_real_data.py
python scripts/04_integrate_bog.py          # builds the canonical cedi-integrated panel
python scripts/02_run_pipeline.py --data real
python scripts/05_publication_analysis.py
python scripts/06_gof_8family.py
python scripts/07_full60_correction.py
python scripts/08_disaggregated_stress.py
python scripts/09_pinksheet_check.py         # needs data/raw/CMO-Historical-Data-Monthly.xlsx
python scripts/10_cedi_alternatives.py
python scripts/11_cedi_hurdle.py
python scripts/12_cedi_mixture.py
python scripts/13_cedi_markov.py

# Compile the paper
cd paper/latex && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Data sources and licensing

Commodity futures (CC=F, GC=F, BZ=F, CL=F) are from Yahoo Finance —
fetch-only under Yahoo's terms, **not committed to this repo**. The
cedi series is the **Bank of Ghana interbank USD/GHS mid-rate**,
official public data, **committed under `data/raw/bog/`** with
attribution — return correlation with the free Yahoo GHS=X proxy is
only 0.18 even after excluding two outright Yahoo data errors (see the
paper, Section 3.1). Monthly CPI and merchandise exports are from the
same BoG tables. World Bank Pink Sheet monthly commodity prices are
used for roll-effect validation only, not redistributed here.

## What's open

See the paper's Limitations section: vine copulas or a nonparametric
copula surface as the natural next step given six of ten pairs reject
every one of eight tested families; a fully family-wise correction
across all three stress-window definitions simultaneously (180 cells,
versus 60 within each definition tested here); and extending the sample
to include Ghana's 2014–2015 currency crisis as a third independent
stress episode.

## License

MIT — see `LICENSE`.
