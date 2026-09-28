# Manuscript Consistency Audit

Branch: `manuscript-publication-revision`

Scope: Introduction, Related Literature, Data and Calendar Construction, Methodology, Results, Discussion, Robustness and Limitations, Conclusion, bibliography keys, figure structure, table structure, and cross-references.

## Frozen evidence checked

- Panel A: 3,008 returns, 2015-01-05 to 2026-07-15, business-day alignment, forward fill capped at three business days.
- Panel B: 2,774 returns, 2015-01-05 to 2026-07-10, consecutive common actual-source dates, no forward-filled return observations.
- The two panels are co-equal.
- Four commodity margins pass the stated adequacy gate in both panels; the Cedi does not.
- Static copula GOF:
  - Gold--Brent, Gold--WTI, Brent--WTI: 8/8 rejected in both panels.
  - Cocoa--Gold: Panel A 8/8 rejected; Panel B 2/8 rejected.
  - Cocoa--Brent: Panel A 8/8 rejected; Panel B 3/8 rejected.
  - Cocoa--WTI: Panel A 8/8 rejected; Panel B 3/8 rejected.
  - All four commodity--Cedi pairs: 0/8 rejected in both panels.
- Stress families use 60 tests each, moving-block bootstrap B=15,000, block length 20.
- Nominal stress counts:
  - Combined: Panel A 6/60, Panel B 5/60.
  - COVID-19: Panel A 13/60, Panel B 11/60.
  - 2024: Panel A 6/60, Panel B 7/60.
- Bonferroni discoveries: 0 in every panel-by-stress family.
- BH-FDR discoveries: 0 in every panel-by-stress family.
- Panel B Cedi POT/GPD at q=0.90: xi≈0.693, beta≈0.329, VaR99≈2.332, ES99≈7.616; interpreted only as a conditional diagnostic because the Cedi marginal is inadequate.
- Panel A PIT-clipping artifact is numerically negligible for the prespecified tail sets and observed GOF statistics.
- Earlier monthly transmission results are excluded from the frozen publication evidence.

## Corrections made during this audit

1. Removed an unsupported Chen et al. (2023) literature statement that had no matching bibliography entry or verified source in the repository.
2. Removed the repository-only study-design workflow from the main manuscript so that the main figure sequence is:
   - Figure 1: calendar workflow
   - Figure 2: finite-threshold concentration
   - Figure 3: static copula GOF
   - Figure 4: stress-summary evidence
3. Replaced the cross-panel summary image in the main manuscript with a native LaTeX table.
4. Kept the study-design workflow as a repository asset and the graphical abstract as a separate submission asset.
5. Kept supplementary diagnostics as Figures S1--S3.
6. Reworded the Panel B Cedi follow-up language so that it reflects the implemented pipeline trigger without claiming undocumented preregistration.
7. Corrected the withdrawn monthly-transmission narrative to match the documented export-extraction audit rather than an unsupported CPI-zero claim.
8. Removed duplicated limitations prose from the Discussion and pointed to the dedicated Robustness and Limitations section.
9. Added proper captions and labels to manuscript tables.
10. Confirmed that all LaTeX citation keys used in the manuscript exist in `references.bib`.
11. Confirmed that all current LaTeX cross-references resolve to defined labels and that no duplicate labels exist.
12. Updated repository documentation to identify `manuscript-publication-revision` as the active publication branch.

## Remaining checks before submission

- Compile the latest branch in Overleaf after pulling the audit commits.
- Inspect the newly compiled PDF for table page breaks, float placement, caption spacing, and supplementary-figure legibility.
- Refresh the verification ledger so that it points to the frozen two-panel publication outputs and current manuscript claims.
- Adapt the final LaTeX package to the target journal's submission template and metadata requirements.
- Keep `paper/latex/paper.tex` and other historical artifacts clearly marked as provenance only.
