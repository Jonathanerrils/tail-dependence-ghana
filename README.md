# Commodity Tail Dependence and Exchange-Rate Risk in Ghana

**Copula adequacy, stress tests, calendar robustness, and reproducible reconstruction of cocoa, gold, Brent, WTI, and Ghana cedi dependence.**

> **Current status:** The scientific reconstruction is substantially complete. The publication analysis now uses two co-equal calendar panels, the publication-resolution stress analysis has been completed for both panels, the major implementation audits are closed, and a reconstructed LaTeX manuscript is available on the `manuscript-rebuild-latex` branch. Historical scripts, outputs, and the older manuscript remain in the repository for provenance and must not be treated as current publication authority unless explicitly identified as verified.

Working manuscript title:

**Commodity Tail Dependence and Exchange-Rate Risk in Ghana: Copula Adequacy, Stress Tests, and Calendar Robustness**

Target journal: *Risks*.

---

## Scientific design

The paper uses two co-equal calendar constructions. Neither is treated as the true, corrected, or primary panel.

### Panel A: business-day alignment

- 3,008 return observations
- 5 January 2015 to 15 July 2026
- business-day calendar
- forward fill limited to three business days
- preserves nominal business-day frequency

Frozen return SHA-256:

```text
3e9769f42ec7925aacd951277ddae6ff529822408815b01210de87659221341f
```

Frozen PIT SHA-256:

```text
be42455c12bf2f4a9c7ef33494930ad396f569192f571386d099046a388119d1
```

### Panel B: common-observation alignment

- 2,774 return observations
- 5 January 2015 to 10 July 2026
- no forward-filled return observations
- every multivariate row uses the same pair of common actual observation dates
- return intervals span one to six calendar days, median one

Frozen return SHA-256:

```text
e6312d6163aeb7c9fb7ce722ebe46be1898f23855d5a87395830e4d1e82bf3b3
```

Frozen PIT SHA-256:

```text
9b322bda7c4e63534da14d1a7e360e758762e98cb39101cf329cdceb5afc216f
```

### Interpretation rule

Results are classified as:

- **calendar-robust** when the same inferential conclusion survives both panels;
- **calendar-sensitive** when the conclusion changes materially across the two constructions;
- **panel-specific diagnostic** when an analysis belongs to only one panel and is not promoted to a cross-panel result.

This classification was frozen before the final stress interpretation.

---

## Marginal-model gate

The publication workflow uses a fixed four-model ladder:

1. AR(1)-GJR-GARCH(1,1)-t
2. AR(2)-GJR-GARCH(1,1)-t
3. AR(1)-EGARCH(1,1)-t
4. AR(1)-GJR-GARCH(1,1)-skew-t

A candidate is admissible only when:

- `convergence_flag == 0`;
- log likelihood and parameters are finite;
- standardized residuals are finite;
- conditional volatility is finite and positive;
- PIT values are finite.

Adequacy then requires all three 5% diagnostics to pass:

- Ljung-Box on standardized residuals;
- Ljung-Box on squared standardized residuals;
- Kolmogorov-Smirnov test of PIT uniformity.

If no valid candidate is adequate, the lowest-AIC valid model is retained only as an **inadequate reference model**.

### Selected margins

| Series | Panel A | Status | Panel B | Status |
|---|---|---|---|---|
| Cocoa | AR(1)-GJR-GARCH-t | Adequate | AR(1)-GJR-GARCH-t | Adequate |
| Gold | AR(1)-GJR-GARCH-t | Adequate | AR(1)-GJR-GARCH-t | Adequate |
| Brent | AR(1)-GJR-GARCH-skew-t | Adequate | AR(1)-GJR-GARCH-t | Adequate |
| WTI | AR(1)-EGARCH-t | Adequate | AR(1)-GJR-GARCH-t | Adequate |
| Cedi | AR(1)-GJR-GARCH-t | Inadequate reference | AR(1)-EGARCH-t | Inadequate reference |

The Cedi remains an explicit modeling limitation under both calendar constructions.

---

## Copula goodness-of-fit

Eight static bivariate copula families are tested:

- Gaussian
- Student-t
- Clayton
- Gumbel
- Frank
- Joe
- survival Clayton
- survival Gumbel

Absolute adequacy is evaluated using a **Rosenblatt-transform Cramér-von Mises statistic with parametric bootstrap**.

Publication settings:

- 10 unordered asset pairs
- 8 families
- 80 pair-family cells per panel
- 50 x 50 evaluation grid
- `B = 2,000` for every cell
- initial p-values in `[0.03, 0.07]` rerun at `B = 5,000`
- copula re-estimated within every bootstrap sample
- Monte Carlo p-value `(exceedances + 1) / (B + 1)`

### Calendar-robust complete rejection

| Pair | Panel A | Panel B |
|---|---:|---:|
| Gold-Brent | 8/8 rejected | 8/8 rejected |
| Gold-WTI | 8/8 rejected | 8/8 rejected |
| Brent-WTI | 8/8 rejected | 8/8 rejected |

### Calendar-robust non-rejection

| Pair | Panel A | Panel B |
|---|---:|---:|
| Cocoa-Cedi | 0/8 rejected | 0/8 rejected |
| Gold-Cedi | 0/8 rejected | 0/8 rejected |
| Brent-Cedi | 0/8 rejected | 0/8 rejected |
| WTI-Cedi | 0/8 rejected | 0/8 rejected |

### Calendar-sensitive cocoa pairs

| Pair | Panel A | Panel B |
|---|---:|---:|
| Cocoa-Gold | 8/8 rejected | 2/8 rejected |
| Cocoa-Brent | 8/8 rejected | 3/8 rejected |
| Cocoa-WTI | 8/8 rejected | 3/8 rejected |

A GOF non-rejection does not prove model correctness. Complete rejection is limited to the eight tested static families and does not imply that copulas in general fail.

---

## Stress-period tail concentration

Finite-threshold lower- and upper-tail concentration is evaluated at:

```text
q = 0.025, 0.05, 0.10
```

for 10 pairs, two tails, and three stress definitions.

Each stress family therefore contains:

```text
10 pairs x 3 thresholds x 2 tails = 60 tests
```

The three families are:

1. COVID-19, 1 March to 30 June 2020;
2. calendar year 2024;
3. combined COVID-19 plus 2024.

The common calm baseline excludes both stress windows.

Inference uses:

- moving-block bootstrap;
- block length 20;
- `B = 15,000` per cell;
- reranking within every regime and bootstrap replicate;
- Bonferroni correction within each 60-test family;
- Benjamini-Hochberg FDR correction within each 60-test family.

### Publication-resolution result

| Stress definition | Panel A nominal p<0.05 | Panel B nominal p<0.05 | Panel A corrected | Panel B corrected |
|---|---:|---:|---:|---:|
| Combined COVID + 2024 | 6/60 | 5/60 | 0/60 | 0/60 |
| COVID-19 only | 11/60 | 11/60 | 0/60 | 0/60 |
| 2024 only | 6/60 | 7/60 | 0/60 | 0/60 |

No cell survives either Bonferroni or Benjamini-Hochberg correction in any of the three families under either panel.

The defensible conclusion is therefore:

> **Within the prespecified stress windows, thresholds, moving-block-bootstrap design, and multiplicity procedures, the absence of adjusted stress discoveries is calendar-robust.**

This is not evidence that tail dependence is constant or that crises cannot affect the markets.

---

## Cedi robustness and EVT

The original Panel A analysis includes seven Cedi treatments:

- baseline GARCH reference;
- raw ranks;
- AR-only filtering;
- weekly aggregation;
- hurdle Student-t;
- hurdle mixture;
- hurdle Markov.

Nominal 5% diagnostic counts out of 12 cells are:

```text
A baseline GARCH     3
B raw ranks          3
C AR-only            2
D weekly             0
E hurdle Student-t   2
F hurdle mixture     2
G hurdle Markov      2
```

These are uncorrected diagnostic results, not headline discoveries.

A synchronized seven-treatment Panel B rerun was prespecified to occur only if at least one commodity-Cedi Panel B stress cell survived Bonferroni or BH correction. No such cell survived, so the trigger was not activated.

For Panel B, the q90 Cedi POT/GPD reference gives approximately:

```text
xi       = 0.693
beta     = 0.329
VaR99    = 2.332
ES99     = 7.616
```

Across thresholds from `q = 0.85` to `0.975`, the Cedi shape estimate remains positive, approximately `0.63` to `0.79`.

These are **conditional diagnostics** because the selected Cedi marginal is inadequate.

---

## Closed implementation audits

### Panel A marginal convergence audit

All 20 Panel A marginal candidates were rechecked with convergence and finite-value validity enforced as hard requirements.

- one Cedi EGARCH candidate was invalid/non-converged;
- no previously selected Panel A marginal changed.

### GOF estimator-equivalence audit

The historical Panel A and publication Panel B GOF fitting routes were tested on identical inputs.

Across all eight families and ten cells:

- fitted parameters matched;
- observed GOF statistics matched;
- maximum absolute observed statistic difference was zero.

No Panel A GOF rerun was required on estimator-equivalence grounds.

### PIT clipping/tie audit

Panel B contains no exact PIT ties.

Panel A contains one three-way Cedi PIT floor tie at `1e-6`. Recovering the underlying residual rank ordering:

- changes only two pseudo-observations;
- changes no membership at `q = 0.025, 0.05, 0.10`;
- changes the largest observed Cedi-pair GOF statistic by only `0.000262`;
- leaves the borderline Cocoa-Cedi Frank observed statistic unchanged.

The clipping issue is therefore closed and is not a material driver of the publication conclusions.

---

## Monthly transmission branch

The previous monthly transmission analysis is **withdrawn from the manuscript**.

The macroeconomic audit identified invalid literal zero values in the extracted Bank of Ghana headline inflation series for part of 2023 and a source gap in the exports series that had previously been masked by the extraction pipeline.

The historical monthly regression coefficients and p-values are not current evidence.

Any future transmission extension must begin with a separately validated macroeconomic data pipeline.

---

## Data sources

### Commodity prices

Daily-source-frequency commodity series:

- Cocoa: `CC=F`
- Gold: `GC=F`
- Brent crude oil: `BZ=F`
- WTI crude oil: `CL=F`

The manuscript does not make unverified claims about the provider's continuous-futures roll or back-adjustment methodology.

### Ghana cedi

The publication exchange-rate source is the Bank of Ghana interbank USD/GHS series.

Returns are oriented so that:

```text
negative Cedi return = depreciation
```

This places adverse commodity and Cedi movements in the same lower-tail orientation.

---

## Reconstructed manuscript

The historical manuscript in `paper/latex/paper.tex` is retained for provenance.

The reconstructed scientific manuscript is being reviewed on the branch:

```text
manuscript-rebuild-latex
```

and is assembled under:

```text
paper/latex/rebuild/
  manuscript.tex
  references.bib
  README.md
  sections/
    01_introduction.tex
    02_related_literature.tex
    03_data_calendar.tex
    04_methodology.tex
    05_results.tex
    06_discussion.tex
    07_robustness_limitations.tex
    08_conclusion.tex
```

Draft pull request:

**PR #1: Rebuild manuscript in LaTeX from audited scientific sections**

The reconstructed manuscript currently contains Sections 1 through 8 and is awaiting final cross-section consistency, bibliography, compilation, figure/table, and submission-format checks before merge.

---

## Repository layout

The publication reconstruction currently uses the following main areas:

```text
paper/
  latex/
    paper.tex                       Historical manuscript
    rebuild/                        Reconstructed LaTeX manuscript
  VERIFICATION_LEDGER.csv           Historical ledger; not current authority

src/tailrisk/
  marginals.py                      Marginal models
  evt.py                            POT/GPD diagnostics
  copulas.py                        Eight-family copula estimation
  gof.py                            GOF implementation
  inference.py                      Bootstrap and diagnostic utilities

scripts/
  00-22                             Historical/original workflow
  23_data_provenance_audit.py
  24_alignment_macro_integrity_audit.py
  25_native_return_panel_audit.py
  26_native_return_panel_corrected_audit.py
  27_synchronized_common_interval_audit.py
  28_publication_rebuild_stage1.py
  29_publication_rebuild_stage2_gof.py
  30_original_panel_convergence_gate_audit.py
  31_publication_rebuild_stage3_stress_SAFE.py
  32_gof_implementation_equivalence_audit.py
  33_pit_ties_audit.py
  34_panelA_cedi_pit_tie_diagnostic.py
  35_panelA_pit_floor_audit_SAFE.py
  36_panelA_cedi_declipped_rank_gof_sensitivity_SAFE_v2.py

outputs/
  tables/                           Panel A and historical/audit outputs
  original_panel_gate_audit/        Panel A convergence audit
  gof_equivalence_audit/            GOF implementation audit
  pit_ties_audit/                    PIT clipping/tie audit
  publication_rebuild_v2/            Panel B publication outputs
```

Older files remain for provenance. Their presence does not make stale results current evidence.

---

## Reproduction status

The project is not yet reduced to one final publication command.

The main publication stages are:

### Panel A hard-gate audit

```bash
python scripts/30_original_panel_convergence_gate_audit.py
```

### Panel B marginals and EVT

```bash
python scripts/28_publication_rebuild_stage1.py
```

### Panel B GOF

```bash
python scripts/29_publication_rebuild_stage2_gof.py
```

### Panel B stress inference

The publication stress analysis has already been completed with the safe Stage 31 workflow. Do not rerun the full `B = 15,000` jobs casually because they are computationally expensive.

### Implementation audits

```bash
python scripts/32_gof_implementation_equivalence_audit.py
python scripts/33_pit_ties_audit.py
python scripts/34_panelA_cedi_pit_tie_diagnostic.py
python scripts/35_panelA_pit_floor_audit_SAFE.py
python scripts/36_panelA_cedi_declipped_rank_gof_sensitivity_SAFE_v2.py
```

A final release workflow will be added after the manuscript, figures, tables, bibliography, and verification ledger are frozen.

---

## Local-versus-GitHub authority

A substantial part of the reconstruction was completed on the author's local PC before being synchronized to GitHub.

Until the local reconstruction is fully uploaded:

> **The audited local project files and verified publication outputs are the authority when they conflict with stale files currently on GitHub.**

Do not infer the current implementation solely from historical GitHub files.

Before the next public release, the local scripts, source modules, publication outputs, manuscript-support files, and documentation will be synchronized into the reconstruction branch and reviewed before merge.

---

## Files that should not be committed

Do not commit Python virtual environments, caches, local IDE state, or transient build artifacts.

In particular, the local project currently contains both `.venv/` and `venv/`. Neither belongs in Git.

Use the repository `.gitignore` and inspect `git status` before committing.

---

## Remaining release work

1. synchronize the current local reconstruction with GitHub;
2. reconcile any differences between local `src/` and stale GitHub source files;
3. upload current scripts 23-36 and publication outputs that form the evidence record;
4. rebuild the current verification ledger from the frozen evidence;
5. run cross-section consistency and citation audits on the assembled manuscript;
6. compile and inspect the LaTeX manuscript;
7. refresh publication tables and figures from frozen outputs;
8. normalize the bibliography;
9. adapt the final paper to *Risks* submission format;
10. create a clean release tag and one-command reproduction workflow.

---

## License

MIT. See `LICENSE`.
