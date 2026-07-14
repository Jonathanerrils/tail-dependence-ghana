# Tail Dependence and Extreme Commodity Risk in Ghana: A Copula-EVT Analysis of Cocoa, Gold and Crude Oil

**Working paper — draft v0.1**

> **Status note.** All results in this draft are estimated on REAL data: daily futures/spot prices (Yahoo Finance/FRED-sourced, 2015-01 to 2026-07) and the Bank of Ghana interbank USD/GHS mid-rate as the primary cedi series. The synthetic validation run that previously occupied §5 is retained as Appendix A (pipeline validation). Every number quoted below appears in a table under `outputs/tables/`; every data-handling decision is logged in `DECISIONS.md`.

## Abstract

Ghana's macro-financial position is unusually concentrated in three commodities: cocoa, gold and crude oil jointly account for the large majority of export receipts. Standard correlation-based risk measures understate the exposure this creates if commodity prices co-move more strongly during extreme market conditions than in normal times. This paper tests that hypothesis directly on real daily data (January 2015 to July 2026): cocoa, gold, Brent and WTI futures filtered through adequacy-gated AR-GJR-GARCH-t (or EGARCH-t) marginals, and the Bank of Ghana interbank USD/GHS mid-rate — not the commonly used but unreliable Yahoo Finance proxy, whose daily returns we show correlate with the official rate at only 0.18 — as the cedi series. Tail dependence is estimated via Gaussian, Student-t, Clayton and Gumbel copulas with a Rosenblatt-transform Cramér-von Mises goodness-of-fit test, a nonparametric empirical estimator, and a rolling 250-day t-copula, with calm-versus-stress comparisons (COVID 2020; the 2024 cocoa shock) assessed by moving-block bootstrap with Bonferroni correction for ten simultaneous pair-tests. We do not find robust evidence that tail dependence among the three commodities intensifies in stress, nor of a detectable commodity-to-cedi transmission channel at daily-to-monthly frequency; the empirical and model-based estimators agree on this null result. The one nominally Bonferroni-significant finding — an apparent decrease in commodity-cedi tail dependence during stress — is resolved, via a standard robustness check excluding the COVID sub-window, as a small-sample artifact rather than a real phenomenon. The most robust pattern in the data is the structural Brent-WTI linkage (empirical λ̂_L 0.72-0.86 throughout), consistent with their status as close substitutes rather than with stress-specific contagion. We report this honestly as a valid null result and provide the complete, reproducible copula-EVT-network pipeline, including the goodness-of-fit and multiple-testing machinery that make the null credible rather than merely an absence of a positive finding.

## 1. Introduction

A country whose export basket is dominated by three commodities does not face three separate price risks; it faces one joint risk whose severity depends on how the three prices behave *together*, and especially how they behave together on bad days. Linear correlation, the workhorse summary of co-movement, is estimated primarily from the center of the joint distribution and can be nearly uninformative about the tails: a Gaussian dependence structure has *zero* asymptotic tail dependence at any correlation below one, so a risk model built on it will mechanically assume that extreme losses arrive one market at a time. If, instead, commodity markets crash together — as the contagion literature suggests they increasingly do — then Ghana's effective exposure is larger than any correlation matrix implies, and the transmission into the cedi, inflation and export revenue is correspondingly understated.

This paper asks two questions. First, do cocoa, gold and crude oil exhibit statistically meaningful lower-tail dependence, and is that dependence stronger during identifiable stress episodes (the COVID shock of March–June 2020 and the historic cocoa price shock of 2024) than in calm periods? Second, does joint commodity tail risk transmit into Ghanaian macro-financial variables, beginning with the cedi and extending, at lower frequency, to inflation and export receipts?

The 2024 cocoa episode makes the question timely and gives it a distinctive twist. Unlike a classic joint crash, 2024 was an *upper-tail* event for cocoa — West African supply failures drove prices to repeated all-time highs — occurring alongside comparatively orderly gold and oil markets. Whether cocoa's extreme behaviour is dependence-generating (dragging other markets and the cedi with it) or idiosyncratic is precisely the kind of question tail-dependence methods can answer and correlation cannot: the Gumbel and survival-Clayton copulas allow upper- and lower-tail asymmetry, and the tail-concentration function can be read separately in each corner.

Our contribution is threefold. Methodologically, we combine a full IFM copula stack with peaks-over-threshold EVT margins and a network representation, applied at the commodity-country nexus rather than to equity markets where such tools are more common. Empirically, we provide (to our knowledge) the first tail-dependence network linking the three commodities that dominate a single African economy's exports to that economy's exchange rate. Practically, the accompanying open-source pipeline produces a reproducible tail-risk monitor — rolling co-crash estimates and stress-versus-calm networks — usable by researchers and policy institutions.

## 2. Related literature

Three strands intersect here. The copula and tail-dependence literature (Sklar 1959; Joe 1997; Nelsen 2006; McNeil, Frey and Embrechts 2015) establishes that dependence beyond correlation is captured by the copula, and that families differ sharply in their tail behaviour: the Gaussian copula is asymptotically tail-independent, the Student-t copula symmetric and tail-dependent, and the Clayton and Gumbel families tail-asymmetric. Hua and Joe (2011) generalize this taxonomy through tail-order concepts that distinguish intermediate tail dependence, which we adopt as an interpretive frame when empirical λ̂_L(q) declines with q. The extreme value strand (Balkema and de Haan 1974; Pickands 1975; Embrechts, Klüppelberg and Mikosch 1997; McNeil and Frey 2000) motivates our GPD treatment of the loss tails and the two-step GARCH-EVT construction. Finally, the commodity-contagion and financialization strand (Tang and Xiong 2012; Silvennoinen and Thorp 2013; Demirer et al. on commodity-emerging market linkages; Reboredo on oil-exchange rate copulas) documents rising cross-commodity co-movement and its transmission into commodity-dependent currencies, though cocoa — a market with a distinctive West African supply geography — remains understudied relative to oil and metals.

## 3. Data

The daily layer comprises ICE cocoa futures (CC=F), COMEX gold (GC=F), Brent (BZ=F) and WTI (CL=F) crude futures from Yahoo Finance, and — critically — the **Bank of Ghana interbank USD/GHS mid-rate** as the cedi series, January 2015 through 10 July 2026 (3,005 aligned business days), in continuously compounded percentage returns. The cedi's returns are sign-flipped so that the lower tail denotes depreciation; thus the lower tail means "bad for Ghana" for every series.

The choice of the official interbank rate over the ubiquitous Yahoo GHS=X proxy is consequential and documented in `outputs/tables/cedi_crosscheck.csv`: while the two series' *levels* agree closely once two outright Yahoo data errors are removed (level correlation 0.997; the errors are a 573.00 print on 2020-03-31 against BoG's 5.44, and a 1.00 print on 2016-05-13 against 3.81), their daily *returns* correlate at only 0.18 — Yahoo's cedi returns are dominated by stale-quote noise and are unusable for tail analysis. The December 2022 crisis illustrates the danger: BoG records the cedi's post-IMF-agreement appreciation as a smooth 12.90→8.00 move over 8–15 December, whereas Yahoo sat stale near 11.8 and then collapsed 33 log-percent in a single artificial catch-up print on 16 December (adjudication in `cedi_crisis_resolution.csv`). No automatic spike-masking is applied to the official series; the largest BoG moves are genuine and retained.

The cedi series carries one structural feature that shapes everything downstream: as a managed float, 22.2% of its daily returns are exactly zero and 38.4% are below 0.01% in magnitude, i.e., the distribution mixes a point mass of administered quiet days with rare, very large adjustments.

The macro layer is monthly from Bank of Ghana tables: headline CPI inflation (available through 2023-12) and merchandise exports f.o.b. (2005-01 to 2023-12). Because macro variables are monthly while the dependence machinery is daily, the transmission analysis proceeds in two stages: daily commodity–cedi tail dependence first, then monthly regressions of depreciation, inflation and export changes on daily tail-risk aggregates.

## 4. Methodology

### 4.1 Marginal models

Copula estimation requires (approximately) i.i.d. uniform margins. Raw commodity returns are neither independent nor identically distributed — they exhibit volatility clustering and leverage effects — so each series is first filtered through an AR(1)-GJR-GARCH(1,1) model with Student-t innovations:

r_t = μ + φ r_{t−1} + ε_t, ε_t = σ_t z_t, z_t ∼ t_ν(0, 1),
σ_t² = ω + (α + γ·1{ε_{t−1}<0}) ε_{t−1}² + β σ_{t−1}².

The standardized residuals ẑ_t = ε̂_t/σ̂_t are mapped to pseudo-uniforms via the fitted t distribution (the probability integral transform), and — as a rank-based safeguard against marginal misspecification — re-ranked into pseudo-observations u_t = rank(ẑ_t)/(n+1) before copula estimation. This is the canonical two-step IFM/pseudo-MLE approach (Joe 2005; Genest, Ghoudi and Rivest 1995).

### 4.2 Extreme value margins

The far loss tail of each standardized-residual series is modelled by peaks-over-threshold: exceedances of the 90th-percentile threshold u are fitted with a generalized Pareto distribution GPD(ξ, β), justified by the Pickands–Balkema–de Haan theorem. The shape ξ governs tail heaviness (ξ > 0: Fréchet-type heavy tail), and POT-implied VaR and ES follow from the standard tail formulas. Because GPD estimates are threshold-sensitive, we report ξ̂ across thresholds from the 85th to the 97.5th percentile (Figure 2) and treat instability across that range as grounds for caution; the Hill estimator on the top 5% order statistics provides a semi-parametric cross-check. Block-maxima GEV fits are a planned robustness layer.

### 4.3 Copulas and tail dependence

For each of the ten pairs we estimate four copulas by pseudo-maximum likelihood: Gaussian (tail-independent benchmark), Student-t (symmetric tail dependence λ_L = λ_U = 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))), Clayton (lower-tail dependence λ_L = 2^{−1/θ}) and Gumbel (upper-tail dependence λ_U = 2 − 2^{1/θ}). Model selection is by AIC. A Gaussian rejection in favour of the t family on a given pair is itself evidence of tail dependence, since the two nest as ν → ∞.

Because fitted-family tail coefficients are only as good as the family assumption, we also compute the nonparametric tail-concentration estimates λ̂_L(q) = P(U ≤ q, V ≤ q)/q at q = 0.05 and its upper-tail analogue, and — following the full-range spirit of Hua (2015) — inspect λ̂_L(q) across q to distinguish genuine asymptotic dependence from intermediate tail dependence that decays toward zero.

### 4.4 Stress analysis and networks

Two stress windows are defined ex ante: the COVID shock (March–June 2020) and the 2024 cocoa shock (calendar 2024). Empirical tail dependence is re-estimated on stress and calm subsamples (with within-subsample re-ranking), and the resulting λ̂_L matrices are rendered as weighted networks whose nodes are the five daily series and whose edge weights are lower-tail dependence. Densification of the stress network relative to the calm network is the visual signature of tail contagion. Dynamics are captured by 250-day rolling-window estimates of λ̂_L(q = 0.10), with ranks recomputed inside each window.

### 4.5 Transmission to Ghana macro variables

The second stage aggregates daily tail information to monthly frequency — the number of joint commodity-loss exceedance days and the month-average rolling λ̂_L — and relates these to cedi depreciation, CPI inflation and export receipts in distributed-lag regressions. This stage is deliberately modest: monthly macro samples are short, and the paper's claims rest primarily on the daily layer.

## 5. Results (real data, 2015-01 to 2026-07)

**Marginals (Table `marginal_garch.csv`).** Cocoa, Gold and WTI pass all
three adequacy diagnostics (Ljung-Box on residuals and squares, KS test
of PIT uniformity, 5% level) after an adequacy-gated search over four
specifications; WTI required EGARCH(1,1)-t rather than the GJR default.
Brent and the cedi fail every specification tried (AR(1)/AR(2)-GJR-
GARCH-t, EGARCH-t, GJR-skew-t) and are retained at best AIC with the
failure disclosed rather than concealed. The cedi's marginal is
intrinsically hard to model with a continuous ARMA-GARCH density: 22.2%
of its daily returns are exactly zero and 38.4% are below 0.01% in
magnitude, consistent with a managed float that holds still for long
stretches and moves in occasional large steps.

**Univariate tails (Table `evt_xi_inference.csv`, Figure 6).** GPD shape
estimates at the 90% threshold are modest and imprecisely bounded for
the commodities (ξ̂: Cocoa 0.047 [-0.08, 0.15], Gold 0.036 [-0.10, 0.13],
Brent 0.083 [-0.05, 0.18], WTI 0.183 [0.05, 0.29] — the only commodity
whose 95% CI excludes zero) and strikingly large for the cedi (ξ̂ =
0.720, 95% CI [0.52, 0.91]), reflecting its rare-but-severe devaluation
episodes. Block-maxima GEV shapes corroborate the ordering (Cocoa 0.071,
Gold 0.150, Brent 0.121, WTI 0.214, Cedi 0.565).

**Copula selection and goodness-of-fit (Tables `tail_dependence.csv`,
`copula_gof.csv`).** AIC prefers the Student-t copula for four of the
five commodity-commodity pairs, but a Rosenblatt-transform Cramér-von
Mises test with a 500-replicate parametric bootstrap — the stage this
draft's previous version lacked — shows that for six of the ten pairs
(all four commodity-commodity pairs plus Brent-WTI) **no family is
rejected at 5%**: the data do not discriminate sharply between Gaussian,
t, Clayton and Gumbel dependence structures at this sample size. For
cedi pairs the picture is mixed: Gold-Cedi, Brent-Cedi and WTI-Cedi have
three or four families simultaneously not rejected (again reflecting
weak power, plausibly linked to the cedi's marginal-adequacy problem
above), while Cocoa-Cedi rejects every family tried — no simple copula
captures that pair's dependence structure.

**Calm versus stress (Tables `calm_stress_lower_qsens_bonferroni.csv`,
Figure 3; upper-tail companion for the 2024 cocoa rally).** Averaged
across all ten pairs, lower-tail dependence is essentially flat from
calm to stress at every quantile tested (q=0.05: 0.145 → 0.149; q=0.10:
0.197 → 0.183; q=0.025: 0.107 → 0.115). Three pairs are nominally
significant at 5% uncorrected, all involving the cedi, and all in the
*decreasing* direction (stress estimate of exactly zero); only WTI-Cedi
survives Bonferroni correction for the ten simultaneous tests. We do
NOT interpret this as evidence that commodity-cedi tail dependence
collapses in stress. The 349-day combined stress window (COVID +
2024) yields only ~17 raw observations per 5% tail, and a
robustness check resolves the anomaly directly: excluding COVID from
the stress definition (leaving 2024 alone) removes significance from
every cedi pair, including the one that survived Bonferroni (Table
`robustness_excl_covid_lower.csv`). We read this as a small-sample
artifact of the COVID sub-window, not a robust finding, and report it
transparently rather than either suppressing it or overselling it.

**Dynamics (Figure 7, Table `rolling_t_copula.csv`).** The rolling
250-day t-copula corroborates the null result from an independent
angle: correlation for Cocoa-Cedi and Brent-Cedi stays close to zero
throughout the sample (ρ ∈ [-0.15, 0.17] and [-0.11, 0.11]
respectively) with no visible stress-driven surge in either the COVID
or 2024 window. The one pair with genuinely strong, stable dependence
throughout is Brent-WTI (ρ ≈ 0.72-0.86 in every window, empirical
λ̂_L 0.72-0.86) — expected of two close substitutes and not evidence of
stress-specific contagion.

**Robustness (Table `robustness_headline.csv`).** The weekly-frequency
rerun (n=598) reproduces the same qualitative pattern as daily data,
including the same COVID-linked cedi anomaly. The exclude-COVID variant
is the one that changes the conclusion, as detailed above. The World
Bank Pink Sheet monthly roll-effect comparison could not be executed
in the analysis environment (no network access to worldbank.org) and
is left as a documented, ready-to-run script for a local rerun.

**Transmission (Table `transmission_regressions.csv`).** Of nine
regressor-target combinations tested (three targets × three lagged
regressors), one is significant at 5% (lagged rolling Cocoa-Brent
λ_t predicting monthly export growth, p=0.011, R²=0.069, n=88). Given
this is one hit among roughly seventy hypothesis tests conducted across
Stages 4-6 of this analysis, we treat it as likely noise rather than a
transmission channel, and do not build any claim on it. Cedi
depreciation and CPI changes show no significant relationship to
commodity joint-tail activity in this sample (R² ≈ 0.001-0.005).

**Headline conclusion.** On this sample, we do not find robust evidence
that cocoa, gold and crude oil exhibit stronger tail dependence during
stress than during calm periods, nor evidence of a commodity-to-cedi
tail-risk transmission channel. The empirical and model-based estimators
agree on this null result, and the one nominally significant finding is
resolved as a small-sample artifact under a natural robustness check.
The strongest, most robust pattern in the data is the structural
Brent-WTI linkage, unrelated to the stress-contagion hypothesis this
paper set out to test.

## 6. Robustness (completed; further extensions noted)

Threshold sensitivity for every series, block-maxima GEV, block-bootstrap
CIs for λ̂_L, weekly-frequency and exclude-COVID reruns, and Bonferroni-
adjusted significance are now implemented and reported in §5. Remaining
extensions: the World Bank Pink Sheet monthly roll-effect comparison
(script-ready, not executed — no network access in this session);
vine-copula estimation of the full five-dimensional dependence; and
time-varying (DCC-copula or GAS) specifications as a more powerful
alternative to the subsample-split test used here.

## 7. Policy discussion

The honest empirical result tempers rather than eliminates the policy
discussion. We did not find evidence, in this sample, that Ghana's three
main export commodities move together more violently in crises than in
calm periods, nor that such joint moves transmit detectably into the
cedi at daily-to-monthly frequency. Three qualified implications follow.
For reserve management, the Brent-WTI linkage confirms that oil-price
risk should be treated as a single joint exposure rather than two
independent ones, but no comparable case is established here for
cocoa-oil or cocoa-gold co-movement. For cocoa-syndication hedging, the
cedi's own extreme-value profile (ξ̂ = 0.72, the heaviest tail in the
panel by a wide margin) is a more clearly evidenced risk than any
commodity-transmission channel, and argues for treating cedi
depreciation risk on its own terms rather than as a derivative of
commodity shocks. For monetary policy, we would caution against
building an early-warning indicator on the commodity-cedi tail linkage
tested here without a larger sample of independent stress episodes:
the one apparent signal in this analysis dissolved under a standard
robustness check, which is itself the useful lesson for anyone building
a similar monitor on similarly short stress-window samples.

## 8. Conclusion

We provide a complete, reproducible copula-EVT framework for measuring
extreme co-movement among Ghana's three dominant export commodities and
its transmission to the cedi. Applied to real daily data from January
2015 to July 2026 with the Bank of Ghana interbank rate as the cedi
series, the framework does not find robust evidence of stress-driven
tail-dependence intensification among cocoa, gold and crude oil, nor of
a commodity-to-cedi transmission channel detectable at this sample
size and these frequencies — a null result corroborated by both
empirical and model-based estimators and surviving a Bonferroni
correction for multiple testing. The one apparent positive finding
(a Bonferroni-significant *decrease* in commodity-cedi tail dependence
during stress) is resolved, via a standard robustness check, as a
small-sample artifact of the COVID sub-window rather than a real
phenomenon. We regard this as a valid and useful finding in its own
right — analogous to a "no outperformance" result in a forecasting
comparison — rather than a failure of the analysis, and the complete
methodological apparatus (adequacy-gated marginals, goodness-of-fit-
tested copulas, bootstrap inference, and multiple robustness checks)
is what makes that null result credible rather than merely an absence
of a positive result.

## References (indicative)

Balkema, A. and de Haan, L. (1974). Residual life time at great age. *Annals of Probability*. · Embrechts, P., Klüppelberg, C. and Mikosch, T. (1997). *Modelling Extremal Events*. Springer. · Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric estimation procedure of dependence parameters. *Biometrika*. · Hua, L. and Joe, H. (2011). Tail order and intermediate tail dependence of multivariate copulas. *JMVA*. · Hua, L. (2015). Tail negative dependence and its applications. *Insurance: Mathematics and Economics*. · Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall. · McNeil, A. and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance*. · McNeil, A., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management*. Princeton. · Nelsen, R. (2006). *An Introduction to Copulas*. Springer. · Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*. · Reboredo, J. (2012). Modelling oil price and exchange rate co-movements. *Journal of Policy Modeling*. · Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. · Tang, K. and Xiong, W. (2012). Index investment and the financialization of commodities. *Financial Analysts Journal*.


---

## Appendix A — Pipeline validation on synthetic data

Before real data was available, the full pipeline (marginals, EVT,
copulas, networks, bootstrap inference) was validated on a synthetic
dataset engineered with known regime-dependent tail dependence, a
simulated 2024 cocoa shock, and a built-in cedi transmission channel
(`scripts/00_generate_synthetic_data.py`). That run confirmed the
machinery recovers known structure: the Student-t copula beat the
Gaussian on all ten pairs by AIC, average empirical lower-tail
dependence rose from λ̂_L ≈ 0.15 (calm) to ≈ 0.34 (stress) — matching
the engineered effect size — and the 2024-window network visibly
densified relative to the calm network. This validation run is
retained in the repository (`--data synthetic`) as a check that the
estimation code is correct; none of its numbers describe the real
Ghanaian economy, and they are superseded entirely by §5 above.
