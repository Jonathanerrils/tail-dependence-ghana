# Frozen Two-Panel Interpretation Protocol
## Commodity Tail Dependence and Exchange-Rate Risk in Ghana

**Status:** Frozen before inspection of the common-observation panel publication-resolution stress results.

## 1. Purpose

The project uses two co-equal, methodologically defensible calendar constructions.

**Panel A: business-day alignment**
- 3,008 returns
- 5 January 2015 to 15 July 2026
- business-day calendar
- forward fill capped at three business days

**Panel B: common-observation alignment**
- 2,774 returns
- 5 January 2015 to 10 July 2026
- no forward-filled return observations
- the same start and end dates for all five series within each multivariate observation

Neither panel is designated primary. Headline claims are based on the intersection of conclusions supported by both panels. Differences are reported as calendar sensitivity rather than resolved by selecting the more favorable result.

## 2. Stress hypotheses

For each panel, stress inference is conducted separately for:

1. combined COVID-19 plus 2024 stress versus calm;
2. COVID-19 only versus the same calm baseline;
3. 2024 only versus the same calm baseline.

The calm baseline excludes both stress windows in all three comparisons.

Each inferential family contains 10 asset pairs, q = 0.025, 0.05 and 0.10, and lower and upper tails. Therefore each family contains 60 tests and the three families contain 180 cells overall.

The three 60-test families are corrected separately. No pooled 180-test correction is used for the stated analysis.

## 3. Estimand

The reported measure is finite-threshold empirical tail concentration, not an asymptotic tail-dependence coefficient.

Lower tail:

`lambda_L(q) = P(U <= q, V <= q) / q`

Upper tail is evaluated analogously using survival ranks.

The regime contrast is:

`D_obs = lambda_stress(q) - lambda_calm(q)`

## 4. Bootstrap inference

Publication-resolution settings are fixed before inspection of Panel B stress results:

- moving-block bootstrap;
- block length = 20;
- calm and stress windows resampled independently;
- observations reranked within regime and within each bootstrap replicate;
- B = 15,000 replicates per test;
- deterministic SHA-256-derived seeds;
- centered, unstudentized two-sided bootstrap-null p-value:

`p = [1 + sum(|D_b - D_obs| >= |D_obs|)] / (B + 1)`

where `D_b` is the bootstrap stress-minus-calm difference.

Percentile bootstrap intervals may be reported descriptively, but corrected inference is determined from the p-values below.

## 5. Multiple-testing rules

For each 60-test family separately:

**Bonferroni**
- alpha = 0.05
- rejection threshold = 0.05 / 60

**Benjamini-Hochberg**
- alpha = 0.05
- applied to all 60 p-values within that stress definition.

Nominal p < 0.05 cells are descriptive and do not override the multiplicity-controlled conclusion.

## 6. Calendar-robustness classification

### Family-level conclusion

A stress-family conclusion is **calendar-robust** when Panel A and Panel B agree on the presence or absence of discoveries under the same correction procedure.

A stress-family conclusion is **calendar-sensitive** when one panel produces one or more discoveries under a correction procedure and the other produces none, or when the substantive corrected discovery set differs.

### Cell-level conclusion

A cell is called **calendar-robust evidence of change** only if it survives the same multiplicity correction in both panels and the estimated stress-minus-calm difference has the same sign in both panels.

A cell is **calendar-sensitive** if its corrected rejection status changes across panels or its estimated direction reverses.

Cells that are nominally significant in one or both panels but fail correction in both are not promoted as robust discoveries.

## 7. Pre-specified Cedi follow-up trigger

The full seven-treatment Cedi A-G analysis will not be rerun on Panel B automatically.

A Panel B A-G rerun is triggered only if at least one commodity-Cedi stress cell survives Bonferroni or BH correction in Panel B, because Panel A currently has no corrected Cedi stress discoveries.

Nominal-only Cedi significance does not trigger the seven-treatment rerun.

## 8. GOF interpretation already fixed

- Gold-Brent, Gold-WTI and Brent-WTI: 8/8 family rejection under both panels, classified as calendar-robust within the tested static family set.
- Cocoa-Cedi, Gold-Cedi, Brent-Cedi and WTI-Cedi: 0/8 family rejection under both panels, classified as calendar-robust non-rejection within the tested static family set.
- Cocoa-Gold, Cocoa-Brent and Cocoa-WTI: rejection patterns differ materially across panels, classified as calendar-sensitive.

Non-rejection does not establish that a copula is correct. Rejection of all eight tested static families does not establish that copulas generally fail.

## 9. Marginal-model interpretation already fixed

Optimizer convergence and finite-fit validity are hard gates. Panel A hard-gate auditing changes none of the previously selected models.

The Cedi has no adequate tested marginal under either panel. Cedi-based copula and EVT results therefore carry an explicit marginal-model limitation.

## 10. Decision discipline after Panel B stress results

After Panel B stress inference is complete:

1. Apply the rules in this document without changing thresholds, windows, q levels, corrections, block length or bootstrap resolution.
2. Compare Panel B with the already verified Panel A publication-resolution results.
3. Report both panel-specific estimates where they differ materially.
4. Promote only cross-panel-stable corrected conclusions to headline findings.
5. Do not redefine a stress window or add/remove tests because of observed significance.
6. Do not rerun Panel B Cedi A-G unless the pre-specified trigger in Section 7 is met.
7. Any new exploratory analysis prompted by the results must be labeled exploratory.

## 11. Monthly transmission

Monthly transmission is outside this protocol. Previous transmission results remain withdrawn until the CPI series is rebuilt from verified source data and the monthly analysis is rerun.

## 12. Freeze statement

This protocol is fixed before inspection of Panel B publication-resolution stress results. Any later change must be documented with its reason and must not be presented as pre-specified.
