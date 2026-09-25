
## Stage 0 — BoG integration (2026-07-12)
- Masked non-positive wti price(s) on [datetime.date(2020, 4, 20)] (log return undefined).
- D1: BoG interbank USD/GHS mid-rate is the primary cedi series (2015-01-02 → 2026-07-10, 2853 obs); Yahoo GHS=X demoted to cross-check only (D3).
- D2: BoG cedi days with |return|>15% reported, NOT masked: [datetime.date(2022, 12, 15)] — official data; adjudicated in cedi_crisis_resolution.csv.
- D5: masked-date adjudication written to outputs/tables/cedi_crisis_resolution.csv: 2020-03-31: Yahoo artifact; BoG governs; 2020-04-01: Yahoo artifact; BoG governs; 2022-12-16: BoG series retains the move on its true dates
- Yahoo GHS=X contains outright garbage prints (level off by >2x vs BoG): 2016-05-13 yahoo=1.00 vs bog=3.81; 2020-03-31 yahoo=573.00 vs bog=5.44 — these alone destroy naive level/return correlations and are the strongest argument for D1.
- Orientation re-verified: cedi returns sign-flipped so the lower tail = depreciation = bad-for-Ghana for all five series; commodity lower tails are price crashes.
- Monthly macro panel (BoG): CPI YoY 1973-01→2023-12, exports 2005-01→2023-12; cedi monthly depreciation kept in raw orientation (+ = depreciation) for Stage 6.

## Stage 0 — BoG integration (2026-07-14)
- Masked non-positive wti price(s) on [datetime.date(2020, 4, 20)] (log return undefined).
- D1: BoG interbank USD/GHS mid-rate is the primary cedi series (2015-01-02 → 2026-07-10, 2853 obs); Yahoo GHS=X demoted to cross-check only (D3).
- D2: BoG cedi days with |return|>15% reported, NOT masked: [datetime.date(2022, 12, 15)] — official data; adjudicated in cedi_crisis_resolution.csv.
- D5: masked-date adjudication written to outputs/tables/cedi_crisis_resolution.csv: 2020-03-31: Yahoo artifact; BoG governs; 2020-04-01: Yahoo artifact; BoG governs; 2022-12-16: BoG series retains the move on its true dates
- Yahoo GHS=X contains outright garbage prints (level off by >2x vs BoG): 2016-05-13 yahoo=1.00 vs bog=3.81; 2020-03-31 yahoo=573.00 vs bog=5.44 — these alone destroy naive level/return correlations and are the strongest argument for D1.
- Orientation re-verified: cedi returns sign-flipped so the lower tail = depreciation = bad-for-Ghana for all five series; commodity lower tails are price crashes.
- BUGFIX: merchandise_exports_only.csv stacks 8 variables long-form (found: ['Merchandise Exports (f.o.b) (Millions of US Dollars)', 'Merchandise Exports Less Imports_Trade Balance (Millions of US Dollars)', 'Merchandise Exports_Cocoa Beans (Millions of US Dollars)', 'Merchandise Exports_Cocoa Products (Millions of US Dollars)', 'Merchandise Exports_Crude Oil (Millions of US Dollars)', 'Merchandise Exports_Gold (Millions of US Dollars)', 'Merchandise Exports_Other Exports (Millions of US Dollars)', 'Merchandise Exports_Timber and Timber Products (Millions of US Dollars)']); the initial extraction took the 'value' column unfiltered, which mixed in 'Merchandise Exports Less Imports_Trade Balance' (legitimately negative) and commodity sub-components. Filtered to 'Merchandise Exports (f.o.b)' only (228 obs). Remaining negative values after filter: 0 (should be 0 for a pure f.o.b. level).
- Monthly macro panel (BoG): CPI YoY 1973-01→2023-12, exports 2005-01→2023-12; cedi monthly depreciation kept in raw orientation (+ = depreciation) for Stage 6.

## Stage 0 — BoG integration (2026-07-14)
- Masked non-positive wti price(s) on [datetime.date(2020, 4, 20)] (log return undefined).
- D1: BoG interbank USD/GHS mid-rate is the primary cedi series (2015-01-02 → 2026-07-10, 2853 obs); Yahoo GHS=X demoted to cross-check only (D3).
- D2: BoG cedi days with |return|>15% reported, NOT masked: [datetime.date(2022, 12, 15)] — official data; adjudicated in cedi_crisis_resolution.csv.
- D5: masked-date adjudication written to outputs/tables/cedi_crisis_resolution.csv: 2020-03-31: Yahoo artifact; BoG governs; 2020-04-01: Yahoo artifact; BoG governs; 2022-12-16: BoG series retains the move on its true dates
- Yahoo GHS=X contains outright garbage prints (level off by >2x vs BoG): 2016-05-13 yahoo=1.00 vs bog=3.81; 2020-03-31 yahoo=573.00 vs bog=5.44 — these alone destroy naive level/return correlations and are the strongest argument for D1.
- Orientation re-verified: cedi returns sign-flipped so the lower tail = depreciation = bad-for-Ghana for all five series; commodity lower tails are price crashes.
- BUGFIX: merchandise_exports_only.csv stacks 8 variables long-form (found: ['Merchandise Exports (f.o.b) (Millions of US Dollars)', 'Merchandise Exports Less Imports_Trade Balance (Millions of US Dollars)', 'Merchandise Exports_Cocoa Beans (Millions of US Dollars)', 'Merchandise Exports_Cocoa Products (Millions of US Dollars)', 'Merchandise Exports_Crude Oil (Millions of US Dollars)', 'Merchandise Exports_Gold (Millions of US Dollars)', 'Merchandise Exports_Other Exports (Millions of US Dollars)', 'Merchandise Exports_Timber and Timber Products (Millions of US Dollars)']); the initial extraction took the 'value' column unfiltered, which mixed in 'Merchandise Exports Less Imports_Trade Balance' (legitimately negative) and commodity sub-components. Filtered to 'Merchandise Exports (f.o.b)' only (228 obs). Remaining negative values after filter: 0 (should be 0 for a pure f.o.b. level).
- BoG export table encodes a reporting gap as literal 0.00 for 8 months (2023-05 to 2023-12) — exports did not actually collapse to zero; these are missing data, not observations. Masked to NaN before use.
- Monthly macro panel (BoG): CPI YoY 1973-01→2023-12, exports 2005-01→2023-04; cedi monthly depreciation kept in raw orientation (+ = depreciation) for Stage 6.

## Data-integrity bug found and fixed during Stage 6 (exports)
- `merchandise_exports_only.csv` stacks 8 BoG variables long-form (f.o.b.
  exports, trade balance, cocoa/gold/timber/crude-oil sub-components).
  The initial Stage 0 extraction took the `value` column unfiltered,
  contaminating the exports series with negative trade-balance rows.
  FIXED: filtered to `Merchandise Exports (f.o.b)` only. Verified 0
  negative values remain (228 obs).
- Separately, the BoG source table itself encodes an 8-month reporting
  gap (2023-05 to 2023-12) as literal `0.00` rather than blank. Exports
  did not collapse to zero; masked to NaN before use (100 valid monthly
  obs remain, down from a falsely-populated 108).
- Net effect: `exports_growth` regression went from R²=NaN (undefined,
  log of non-positive values) to a well-posed regression (n=88, R²=0.069).

## Interpretation of the Stage 4 Bonferroni-significant result (IMPORTANT)
- At q=0.05, WTI–Cedi is the only pair whose stress-vs-calm tail-dependence
  DECREASE survives Bonferroni correction (lambda: 0.038 calm -> 0.000
  stress). Two more cedi pairs (Gold–Cedi, Brent–Cedi) are significant
  at 5% uncorrected.
- Diagnosis: the 349-day combined stress window (COVID + 2024) yields
  only ~17 raw observations per 5% tail. The stress-window point
  estimate of exactly 0 reflects zero co-exceedances in a 17-vs-17
  comparison — a small-sample artifact of the empirical tail-
  concentration estimator, not evidence of negative dependence.
  Confirmed: pseudo-observations show no ties (349 unique values), so
  this is a genuine small-n draw, not a data-encoding bug.
- Robustness check resolves it: excluding COVID from the stress
  definition (stress = 2024 only) removes ALL cedi-pair significance,
  including Brent–Cedi and WTI–Cedi. The apparent "collapse" was
  COVID-window-specific and does not survive a different, still
  reasonable, stress-window choice.
- CONCLUSION FOR THE PAPER: do not report "tail dependence with the
  cedi decreases in stress" as a finding. Report it as a small-sample
  artifact identified and explained via a robustness check, and note
  q=0.05 with ~350-day stress windows has limited power throughout.
- The rolling 250-day t-copula (Stage 4b) corroborates the null result
  from a different angle: rho for Cocoa-Cedi and Brent-Cedi stays near
  zero (-0.15 to +0.17) in BOTH calm and stress windows — no visible
  stress-driven correlation surge for the cedi pairs.

## Marginal adequacy failures (real data)
- Brent and Cedi fail all four ladder specifications (AR(1)/AR(2)-GJR-
  GARCH-t, EGARCH-t, GJR-skew-t) at the 5% level; best-AIC AR-GJR-
  GARCH-t retained for both, with the failure disclosed in the paper's
  limitations. Cocoa, Gold, WTI pass (WTI via EGARCH-t).
- The cedi's marginal is intrinsically difficult: 22.2% exact-zero
  daily returns and 38.4% |return|<0.01%, consistent with a managed
  float with periods of administered stability punctuated by rare,
  large adjustments. No standard ARMA-GARCH density captures a point
  mass at zero; this is disclosed rather than patched.

## Headline empirical result
- Only Brent-WTI shows strong, high tail dependence throughout (0.72-0.86
  both regimes) -- expected given they are close substitutes, not
  evidence of stress-specific contagion.
- Across all other 9 pairs and q in {0.025, 0.05, 0.10}, both tails: NO
  robust, Bonferroni-significant evidence that tail dependence increases
  in stress. This null result is corroborated by the rolling t-copula.
- Three of four AIC-winning copula families are NOT rejected by the
  Rosenblatt-CvM goodness-of-fit test for cedi pairs (Cocoa-Cedi is the
  exception: no family passes), meaning several parametric shapes remain
  statistically compatible with the cedi's dependence structure --
  another symptom of the marginal-adequacy problem above, not a sign of
  strong asymmetric tail behaviour.

## Post-review methodological expansion (this update)
- Copula family set expanded from 4 to 8 (added Frank, Joe, survival
  Clayton, survival Gumbel). Result: the six pairs that rejected every
  family under the 4-family set still reject every family under the
  8-family set (0/8 for all six); AIC-preferred family changed for
  Cocoa-Brent and Cocoa-WTI (now survival Gumbel) but goodness-of-fit
  rejection is unchanged for both.
- Multiple-testing correction redone properly: full family-wise
  Bonferroni across all 60 cells (previously per-(q,tail)-slice, m=10)
  plus an independent Benjamini-Hochberg FDR correction. Result: 0/60
  survive either correction, for the combined stress window and for
  COVID-only and 2024-only tested separately. One methodological find:
  an initial 500-replicate bootstrap pass on COVID-only flagged 6 cells
  as Bonferroni-significant, all at the exact resolution floor a
  500-replicate bootstrap can produce (p=2/501); re-estimation at 5,000
  and 15,000 replicates resolved all 6 to non-significant.
- World Bank Pink Sheet roll-effect check executed (previously disclosed
  as not-executed due to no network access in the analysis sandbox).
  Result, properly matched to Pink Sheet's monthly-average methodology:
  return correlations 0.98 (Brent), 0.99 (WTI), 0.94 (cocoa), 0.997
  (gold) -- ruling out roll contamination as a material concern. A first
  attempt using Yahoo's month-end snapshot (methodologically mismatched
  against Pink Sheet's monthly average) gave misleadingly low
  correlations (0.62-0.76), corrected before reporting.
- Cedi marginal robustness extended from the original adequacy-gated
  GARCH fit to 6 total treatments: GARCH baseline, unfiltered raw ranks
  (continuization), ARMA-only (no GARCH), weekly frequency, an explicit
  hurdle/zero-inflated model (dynamic AR(2) logistic zero-probability +
  continuous submodel on nonzero returns), and a 2-state Markov-
  switching extension of the continuous submodel (one-step-ahead
  predictive regime probabilities, avoiding look-ahead contamination).
  None of the three hurdle-based treatments achieves full KS uniformity;
  a diagnostic check found why (12.7% of nonzero cedi returns are
  themselves within 0.5% in magnitude -- a dense near-zero cluster
  beyond the exact-zero point mass). Substantive finding: Gold-Cedi's
  significance under the GARCH baseline (p=0.036) is not present under
  any of the other 5 treatments (p=0.08-1.00); Brent-Cedi is a
  consistent borderline case, strengthening under Markov-switching
  (p=0.004, confirmed stable at 15,000 replicates, not a resolution
  artifact); WTI-Cedi and Cocoa-Cedi are stable across all 6 treatments.
- Stale, superseded Stage-0 script `scripts/04_build_real_panel.py`
  removed from the repository (identified during LaTeX verification as
  an earlier, pre-bugfix version left alongside the corrected
  `04_integrate_bog.py`; its orphaned output tables
  `bog_yahoo_crosscheck.csv`, `bog_yahoo_crosscheck_disagreements.csv`,
  `crisis_date_resolution.csv` were never used by any downstream script
  and are not carried forward).
- Full LaTeX manuscript added at `paper/latex/`, reframed around
  financial risk management and model risk (target: *Risks*).
  `paper/paper.md` retained only as a superseded historical draft.

## Stage 0 � BoG integration (2026-08-19)
- Masked non-positive wti price(s) on [datetime.date(2020, 4, 20)] (log return undefined).

## Stage 0 � BoG integration (2026-08-19)
- Masked non-positive wti price(s) on [datetime.date(2020, 4, 20)] (log return undefined).
- D1: BoG interbank USD/GHS mid-rate is the primary cedi series (2015-01-02 → 2026-07-10, 2853 obs); Yahoo GHS=X demoted to cross-check only (D3).
- D2: BoG cedi days with |return|>15% reported, NOT masked: [datetime.date(2022, 12, 15)] — official data; adjudicated in cedi_crisis_resolution.csv.
- D5: masked-date adjudication written to outputs/tables/cedi_crisis_resolution.csv: 2020-03-31: Yahoo artifact; BoG governs; 2020-04-01: Yahoo artifact; BoG governs; 2022-12-16: BoG series retains the move on its true dates
- Yahoo GHS=X contains outright garbage prints (level off by >2x vs BoG): 2016-05-13 yahoo=1.00 vs bog=3.81; 2020-03-31 yahoo=573.00 vs bog=5.44 — these alone destroy naive level/return correlations and are the strongest argument for D1.
- Orientation re-verified: cedi returns sign-flipped so the lower tail = depreciation = bad-for-Ghana for all five series; commodity lower tails are price crashes.
- BUGFIX: merchandise_exports_only.csv stacks 8 variables long-form (found: ['Merchandise Exports (f.o.b) (Millions of US Dollars)', 'Merchandise Exports Less Imports_Trade Balance (Millions of US Dollars)', 'Merchandise Exports_Cocoa Beans (Millions of US Dollars)', 'Merchandise Exports_Cocoa Products (Millions of US Dollars)', 'Merchandise Exports_Crude Oil (Millions of US Dollars)', 'Merchandise Exports_Gold (Millions of US Dollars)', 'Merchandise Exports_Other Exports (Millions of US Dollars)', 'Merchandise Exports_Timber and Timber Products (Millions of US Dollars)']); the initial extraction took the 'value' column unfiltered, which mixed in 'Merchandise Exports Less Imports_Trade Balance' (legitimately negative) and commodity sub-components. Filtered to 'Merchandise Exports (f.o.b)' only (228 obs). Remaining negative values after filter: 0 (should be 0 for a pure f.o.b. level).
- BoG export table encodes a reporting gap as literal 0.00 for 8 months (2023-05 to 2023-12) — exports did not actually collapse to zero; these are missing data, not observations. Masked to NaN before use.
- Monthly macro panel (BoG): CPI YoY 1973-01→2023-12, exports 2005-01→2023-04; cedi monthly depreciation kept in raw orientation (+ = depreciation) for Stage 6.

## Combined-stress publication run (15,000 bootstrap replicates)

Ran scripts/15_publication_stress_combined.py on two independent machines
(Windows, Linux), using the corrected centered-bootstrap p-value in
inference.py and deterministic SHA-256 seeds. Confirmed input identical
via _pit_real.csv comparison (cocoa/gold/wti differ by <1e-6, brent by
<4e-6 -- floating-point noise; cedi differs substantially, consistent
with the already-documented cross-environment instability).

Non-cedi cells reproduced exactly across machines (e.g. Cocoa-Brent
p=0.030931 on both). Cedi-involving cells diverged as expected.

Result: 0/60 Bonferroni, 0/60 BH-FDR on both machines. Minimum p-value
0.0065 (Linux) / 0.030931 (Windows) -- both well above the Bonferroni
threshold of 0.000833.

Output file: outputs/tables/calm_stress_full60_B15000.csv
SHA-256 (Windows run): D26AC380AA5B7B3685351DBEAFFBF4E6F33E8BB6E69493AA131DB7EE75C87FB7

## Publication-resolution stress inference - final adjudication (2026-08-27)

- The stress-dependence inference was rerun using the corrected two-sided
  centered moving-block bootstrap test. The bootstrap distribution of the
  calm-stress difference is centered on the observed difference before
  evaluating the null H0: difference = 0.

- Nominal 5% significance, Bonferroni significance, and Benjamini-Hochberg
  FDR decisions are all derived directly from the centered bootstrap
  p-value. Percentile bootstrap intervals are retained only as descriptive
  uncertainty intervals and are not used for multiple-testing decisions.

- Publication resolution is 15,000 bootstrap replicates per cell.

- Combined COVID + 2024 stress definition:
  60 tests = 10 pairs x 3 q values x 2 tails.
  Uncorrected: 6/60.
  Bonferroni, m=60: 0/60.
  Benjamini-Hochberg FDR, 5%: 0/60.
  Minimum p-value: 0.030931.

- COVID-only stress definition:
  60 tests.
  Uncorrected: 13/60.
  Bonferroni, m=60: 0/60.
  Benjamini-Hochberg FDR, 5%: 0/60.
  Minimum p-value: 0.002000.

- 2024-only stress definition:
  60 tests.
  Uncorrected: 6/60.
  Bonferroni, m=60: 0/60.
  Benjamini-Hochberg FDR, 5%: 0/60.
  Minimum p-value: 0.026132.

- Accordingly, none of the three separately corrected 60-test families
  contains a Bonferroni or BH-FDR discovery. Across the 180 tested cells,
  there are therefore zero corrected discoveries, although correction was
  performed separately within each 60-test stress-definition family rather
  than as one pooled m=180 family.

### Brent-WTI COVID diagnostic

- The unusually large COVID-only Brent-WTI upper-tail difference occurs at
  q=0.05, not q=0.10:
  lambda_calm = 0.729872,
  lambda_COVID = 0.229885,
  difference = -0.499987.

- The result is produced by sparse empirical tail counts in the 87-observation
  COVID window. At q=0.05 there is only one joint Brent-WTI upper-tail event,
  giving 1 / (87 x 0.05) = 0.229885.

- The sole q=0.05 joint upper-tail date is 2020-04-02. The negative WTI price
  observation on 2020-04-20 is not the source of this result. Removing
  2020-04-20 alone leaves the estimate essentially unchanged.

- Threshold sensitivity is substantial. At q=0.10, where six COVID joint
  upper-tail observations are available, the estimated difference falls to
  -0.043979.

- The Brent-WTI q=0.05 COVID result is therefore treated as a sparse-window,
  threshold-sensitive empirical-tail estimate rather than evidence of a
  structural change in Brent-WTI dependence. It does not survive either
  Bonferroni or BH-FDR correction.

### Paper conclusion

The stress analysis finds no robust statistical evidence that commodity
tail dependence intensified during the COVID-19 and 2024 stress periods
examined. Nominal results are reported as sensitivity findings where useful,
but none survives the prespecified multiple-testing corrections.