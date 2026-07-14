
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
