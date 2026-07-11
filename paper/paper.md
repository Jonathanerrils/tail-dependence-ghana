# Tail Dependence and Extreme Commodity Risk in Ghana: A Copula-EVT Analysis of Cocoa, Gold and Crude Oil

**Working paper — draft v0.1**

> **Status note.** The methodology, code and pipeline in this draft are complete and tested. All numerical results currently reported come from a *synthetic validation dataset* engineered to reproduce the stylized facts under study (volatility clustering, regime-dependent tail dependence, the 2024 cocoa episode, and a cedi transmission channel). They demonstrate that the estimation machinery recovers known structure; they are not yet empirical findings. Rerun `scripts/01_fetch_real_data.py` and `scripts/02_run_pipeline.py --data real`, then replace the tables and figures before circulating.

## Abstract

Ghana's macro-financial position is unusually concentrated in three commodities: cocoa, gold and crude oil jointly account for the large majority of export receipts. Standard correlation-based risk measures understate the exposure this creates if commodity prices co-move more strongly during extreme market conditions than in normal times. This paper measures that asymmetry directly. We filter daily returns on cocoa, gold, Brent and WTI futures and the Ghana cedi through AR(1)-GJR-GARCH(1,1) models with Student-t innovations, model the loss tails of the standardized residuals with peaks-over-threshold generalized Pareto distributions, and estimate Gaussian, Student-t, Clayton and Gumbel copulas on the probability-integral-transformed residuals. Tail dependence is measured three ways — analytically from the fitted copulas, nonparametrically via the empirical tail-concentration function, and dynamically through rolling-window co-crash estimates — and summarized as a lower-tail-dependence network contrasting calm periods with the 2020 COVID shock and the 2024 cocoa price shock. In the validation run, the Student-t copula dominates the Gaussian on every pair by AIC, average lower-tail dependence more than doubles from calm to stress periods (λ̂_L = 0.15 to 0.34), and the cedi's tail linkage to commodity co-crashes strengthens markedly in the 2024 window — the pattern the real-data analysis will test. We discuss implications for Ghanaian reserve management, cocoa-syndication hedging and inflation risk.

## 1. Introduction

A country whose export basket is dominated by three commodities does not face three separate price risks; it faces one joint risk whose severity depends on how the three prices behave *together*, and especially how they behave together on bad days. Linear correlation, the workhorse summary of co-movement, is estimated primarily from the center of the joint distribution and can be nearly uninformative about the tails: a Gaussian dependence structure has *zero* asymptotic tail dependence at any correlation below one, so a risk model built on it will mechanically assume that extreme losses arrive one market at a time. If, instead, commodity markets crash together — as the contagion literature suggests they increasingly do — then Ghana's effective exposure is larger than any correlation matrix implies, and the transmission into the cedi, inflation and export revenue is correspondingly understated.

This paper asks two questions. First, do cocoa, gold and crude oil exhibit statistically meaningful lower-tail dependence, and is that dependence stronger during identifiable stress episodes (the COVID shock of March–June 2020 and the historic cocoa price shock of 2024) than in calm periods? Second, does joint commodity tail risk transmit into Ghanaian macro-financial variables, beginning with the cedi and extending, at lower frequency, to inflation and export receipts?

The 2024 cocoa episode makes the question timely and gives it a distinctive twist. Unlike a classic joint crash, 2024 was an *upper-tail* event for cocoa — West African supply failures drove prices to repeated all-time highs — occurring alongside comparatively orderly gold and oil markets. Whether cocoa's extreme behaviour is dependence-generating (dragging other markets and the cedi with it) or idiosyncratic is precisely the kind of question tail-dependence methods can answer and correlation cannot: the Gumbel and survival-Clayton copulas allow upper- and lower-tail asymmetry, and the tail-concentration function can be read separately in each corner.

Our contribution is threefold. Methodologically, we combine a full IFM copula stack with peaks-over-threshold EVT margins and a network representation, applied at the commodity-country nexus rather than to equity markets where such tools are more common. Empirically, we provide (to our knowledge) the first tail-dependence network linking the three commodities that dominate a single African economy's exports to that economy's exchange rate. Practically, the accompanying open-source pipeline produces a reproducible tail-risk monitor — rolling co-crash estimates and stress-versus-calm networks — usable by researchers and policy institutions.

## 2. Related literature

Three strands intersect here. The copula and tail-dependence literature (Sklar 1959; Joe 1997; Nelsen 2006; McNeil, Frey and Embrechts 2015) establishes that dependence beyond correlation is captured by the copula, and that families differ sharply in their tail behaviour: the Gaussian copula is asymptotically tail-independent, the Student-t copula symmetric and tail-dependent, and the Clayton and Gumbel families tail-asymmetric. Hua and Joe (2011) generalize this taxonomy through tail-order concepts that distinguish intermediate tail dependence, which we adopt as an interpretive frame when empirical λ̂_L(q) declines with q. The extreme value strand (Balkema and de Haan 1974; Pickands 1975; Embrechts, Klüppelberg and Mikosch 1997; McNeil and Frey 2000) motivates our GPD treatment of the loss tails and the two-step GARCH-EVT construction. Finally, the commodity-contagion and financialization strand (Tang and Xiong 2012; Silvennoinen and Thorp 2013; Demirer et al. on commodity-emerging market linkages; Reboredo on oil-exchange rate copulas) documents rising cross-commodity co-movement and its transmission into commodity-dependent currencies, though cocoa — a market with a distinctive West African supply geography — remains understudied relative to oil and metals.

## 3. Data

The daily layer comprises ICE cocoa futures (CC=F), COMEX gold (GC=F), Brent (BZ=F) and WTI (CL=F) crude futures, and the USD/GHS exchange rate, from January 2015 through June 2026, in continuously compounded percentage returns. WTI serves as a robustness alternative to Brent. The macro layer is monthly: Ghana CPI inflation and export receipts from Bank of Ghana time-series workbooks, with World Bank series as cross-checks. Because macro variables are monthly while the dependence machinery operates on daily returns, the transmission analysis proceeds in two stages: daily commodity-cedi tail dependence first, then monthly regressions of inflation and export-revenue changes on daily tail-risk aggregates (counts of joint exceedance days, mean rolling λ̂_L within the month).

*(Validation run: synthetic daily data with the schema above, generated by `scripts/00_generate_synthetic_data.py`; see the status note.)*

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

## 5. Results (validation run — synthetic data)

**Marginals (Table `marginal_garch.csv`).** All five series show the expected GARCH persistence (α + β near 0.95) with Student-t degrees of freedom between roughly 4.5 and 7, confirming heavy-tailed innovations even after volatility filtering.

**Univariate tails (Table `evt_pot_gpd.csv`, Figure 2).** GPD shape estimates are positive for oil and the cedi (ξ̂ ≈ 0.11) and near zero for cocoa and gold at the 90% threshold, with the cocoa threshold-sensitivity scan stable across the 85–97.5% range — the property the real-data run must replicate before POT-based VaR is trusted.

**Copula selection and tail dependence (Tables `copula_fits.csv`, `tail_dependence.csv`).** The Student-t copula minimizes AIC on all ten pairs, decisively rejecting the tail-independent Gaussian benchmark. Fitted-t and empirical lower-tail coefficients agree closely, with the Brent–WTI pair strongest (as expected of near-substitutes) and the commodity-cedi pairs showing economically meaningful λ̂_L.

**Calm versus stress (Table `calm_vs_stress_tail_dependence.csv`, Figures 3 and 5).** Average empirical lower-tail dependence rises from 0.15 in calm periods to 0.34 in stress windows. The 2024-window network is visibly denser than the calm network, with the cocoa–cedi and oil–cedi edges strengthening most — the transmission channel of interest.

**Dynamics (Figure 4).** Rolling λ̂_L estimates rise sharply into both shaded stress windows and decay afterwards, demonstrating that the 250-day estimator resolves regime shifts at the horizon relevant for risk monitoring.

The validation run therefore confirms that the pipeline recovers engineered tail structure with the correct sign, ordering and timing. Empirical conclusions await the real-data rerun.

## 6. Robustness (planned)

Threshold sensitivity for every series (not only cocoa); q-sensitivity of λ̂_L over q ∈ {0.01, …, 0.10}; block-maxima GEV as an alternative to POT; WTI-for-Brent substitution; subsample stability excluding COVID; block-bootstrap confidence intervals for λ̂_L in the calm/stress comparison; and, as an extension, vine-copula estimation of the full five-dimensional dependence and time-varying (DCC-copula or GAS) specifications.

## 7. Policy discussion

Three implications follow if the real-data results resemble the validation pattern. For reserve management, tail-dependent commodity receipts mean diversification benefits evaporate exactly when needed, arguing for stress buffers calibrated to joint rather than marginal commodity VaR. For cocoa-syndication hedging, upper-tail asymmetry in cocoa (a Gumbel-type 2024 signature) changes the optimal hedge from symmetric futures positions toward option structures. For monetary policy, a strengthening commodity-cedi lower-tail edge is an early-warning indicator: joint commodity stress days that historically preceded depreciation episodes can be monitored in near-real time with the rolling estimator in Figure 4.

## 8. Conclusion

We provide a complete, reproducible copula-EVT framework for measuring extreme co-movement among Ghana's three dominant export commodities and its transmission to the cedi, with stress-versus-calm tail networks as the summarizing object. The validation run demonstrates the machinery; the real-data application will determine whether Ghana's commodity exposure is, in the tails, materially larger than correlation suggests.

## References (indicative)

Balkema, A. and de Haan, L. (1974). Residual life time at great age. *Annals of Probability*. · Embrechts, P., Klüppelberg, C. and Mikosch, T. (1997). *Modelling Extremal Events*. Springer. · Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric estimation procedure of dependence parameters. *Biometrika*. · Hua, L. and Joe, H. (2011). Tail order and intermediate tail dependence of multivariate copulas. *JMVA*. · Hua, L. (2015). Tail negative dependence and its applications. *Insurance: Mathematics and Economics*. · Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall. · McNeil, A. and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance*. · McNeil, A., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management*. Princeton. · Nelsen, R. (2006). *An Introduction to Copulas*. Springer. · Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*. · Reboredo, J. (2012). Modelling oil price and exchange rate co-movements. *Journal of Policy Modeling*. · Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. · Tang, K. and Xiong, W. (2012). Index investment and the financialization of commodities. *Financial Analysts Journal*.
