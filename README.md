# Commodity Tail Dependence and Exchange-Rate Risk in Ghana: Copula Adequacy, Stress Tests, and Calendar Robustness

**Research repository for a copula-EVT study of cocoa, gold, crude oil and the Ghana cedi, with explicit attention to calendar construction, marginal-model adequacy, stress-period tail concentration and reproducibility.**

> **Project status:** The scientific analysis is under active reconstruction from audited data and reproducible outputs. Some older manuscript claims and pipeline outputs remain in the repository for history and comparison, but they should not be treated as the final publication evidence unless they are identified below as verified.

Working manuscript title:

**Commodity Tail Dependence and Exchange-Rate Risk in Ghana: Copula Adequacy, Stress Tests, and Calendar Robustness**

Target journal: *Risks*.

---

## Why this repository is being rebuilt

The original analysis used a business-day panel with capped forward filling. A later provenance audit showed that this documented rule has material consequences for a study of contemporaneous dependence:

- the Bank of Ghana source ends on **10 July 2026**, while the original processed panel extends to **15 July 2026** through the stated forward-fill rule;
- cross-market calendar alignment creates artificial zero returns on some source-missing or market-closure dates;
- the April 2020 negative WTI settlement requires special treatment because logarithmic returns cannot pass through a non-positive price;
- the Ghana cedi remains unusually concentrated near zero even after calendar-generated zeros are reduced;
- some copula goodness-of-fit conclusions, especially for cocoa-related commodity pairs, change under a stricter common-observation construction.

These findings do **not** mean that the original panel was fabricated or that its documented methodology was invalid. They show that two defensible calendar constructions answer slightly different empirical questions.

For that reason, the publication analysis now uses a **deliberate two-panel design**.

---

## Two-panel design

### Panel A: business-day alignment

This is the project's original documented construction.

- **3,008 return observations**
- **5 January 2015 to 15 July 2026**
- business-day calendar
- forward fill capped at three business days
- preserves nominal business-day frequency
- retained as a fully audited analysis specification

The frozen return file has SHA-256:

```text
3e9769f42ec7925aacd951277ddae6ff529822408815b01210de87659221341f
```

A strengthened convergence audit of all 20 marginal candidates confirmed that **none of the previously selected Panel A marginal models changes** when optimizer convergence and finite-fit validity are imposed as hard requirements.

### Panel B: common-observation alignment

This is the stricter synchronization specification.

- **2,774 return observations**
- **5 January 2015 to 10 July 2026**
- no forward-filled return observations
- each multivariate row uses the same start and end observation dates for all five series
- return intervals range from one to six calendar days
- median interval is one calendar day

The synchronized return file has SHA-256:

```text
e6312d6163aeb7c9fb7ce722ebe46be1898f23855d5a87395830e4d1e82bf3b3
```

### Interpretation rule

Neither panel is treated as the automatic "truth" and neither is described as a correction of an erroneous dataset.

Headline conclusions are classified as:

- **calendar-robust**: the inferential conclusion survives both panel constructions;
- **calendar-sensitive**: the conclusion changes materially across the two constructions;
- **panel-specific diagnostic**: the analysis has only been completed under one construction and is not promoted as a cross-panel result.

This structure was adopted before running any further synchronized-panel stress analysis.

---

## Verified findings so far

### Marginal-model validity

The publication workflow now treats optimizer convergence as a hard admissibility condition.

A candidate marginal model is considered valid only if:

- `convergence_flag == 0`;
- log likelihood and fitted parameters are finite;
- standardized residuals are finite;
- conditional volatility is finite and strictly positive;
- PIT values are finite.

Among valid candidates, the first specification in the fixed ladder that passes the 5% adequacy gate is selected. If none passes, the lowest-AIC valid model is retained only as an **inadequate deterministic reference**.

Candidate ladder:

1. AR(1)-GJR-GARCH(1,1)-t
2. AR(2)-GJR-GARCH(1,1)-t
3. AR(1)-EGARCH(1,1)-t
4. AR(1)-GJR-GARCH(1,1)-skew-t

Panel A selections:

| Series | Selected specification | Status |
|---|---|---|
| Cocoa | AR(1)-GJR-GARCH-t | Adequate |
| Gold | AR(1)-GJR-GARCH-t | Adequate |
| Brent | AR(1)-GJR-GARCH-skew-t | Adequate |
| WTI | AR(1)-EGARCH-t | Adequate |
| Cedi | AR(1)-GJR-GARCH-t | Inadequate reference |

Panel B selections:

| Series | Selected specification | Status |
|---|---|---|
| Cocoa | AR(1)-GJR-GARCH-t | Adequate |
| Gold | AR(1)-GJR-GARCH-t | Adequate |
| Brent | AR(1)-GJR-GARCH-t | Adequate |
| WTI | AR(1)-GJR-GARCH-t | Adequate |
| Cedi | AR(1)-EGARCH-t | Inadequate reference |

The Panel B Cedi reference converges, but its PIT uniformity test fails strongly. The Cedi marginal therefore remains an explicit limitation rather than being forced into an "adequate" specification.

---

## Copula goodness-of-fit

Eight static bivariate copula families are evaluated:

- Gaussian
- Student-t
- Clayton
- Gumbel
- Frank
- Joe
- survival Clayton
- survival Gumbel

AIC is used only for relative comparison. Absolute adequacy is assessed with a Rosenblatt-transform Cramér-von Mises-type parametric-bootstrap test.

Publication GOF settings:

- 10 asset pairs
- 8 copula families
- 80 pair-family cells
- 50 x 50 evaluation grid
- `B = 2,000` bootstrap replicates for every cell
- initial p-values in `[0.03, 0.07]` rerun from scratch at `B = 5,000`
- model re-estimated within each bootstrap sample
- Monte Carlo p-value `(exceedances + 1) / (B + 1)`

### Calendar-robust GOF findings

The following results currently survive both calendar constructions:

| Pair | Panel A | Panel B | Classification |
|---|---:|---:|---|
| Gold-Brent | 8/8 rejected | 8/8 rejected | Calendar-robust |
| Gold-WTI | 8/8 rejected | 8/8 rejected | Calendar-robust |
| Brent-WTI | 8/8 rejected | 8/8 rejected | Calendar-robust |
| Cocoa-Cedi | 0/8 rejected | 0/8 rejected | Calendar-robust |
| Gold-Cedi | 0/8 rejected | 0/8 rejected | Calendar-robust |
| Brent-Cedi | 0/8 rejected | 0/8 rejected | Calendar-robust |
| WTI-Cedi | 0/8 rejected | 0/8 rejected | Calendar-robust |

### Calendar-sensitive GOF findings

The cocoa-related commodity pairs are sensitive to calendar construction:

| Pair | Panel A | Panel B |
|---|---:|---:|
| Cocoa-Gold | 8/8 rejected | 2/8 rejected |
| Cocoa-Brent | 8/8 rejected | 3/8 rejected |
| Cocoa-WTI | 8/8 rejected | 3/8 rejected |

The repository therefore no longer treats "all commodity-commodity pairs reject all eight static copulas" as a universal finding.

A GOF non-rejection does not prove that a copula is correctly specified, and rejection of all eight tested families does not imply that copulas in general fail.

---

## Stress-period tail concentration

The stress design evaluates empirical finite-tail concentration at:

```text
q = 0.025, 0.05, 0.10
```

for:

- 10 asset pairs;
- lower and upper tails;
- three stress definitions.

Each stress definition therefore contains:

```text
10 pairs x 3 q levels x 2 tails = 60 tests
```

The three inferential families are:

1. COVID-19 stress, March-June 2020;
2. calendar year 2024 cocoa-stress window;
3. combined COVID-19 + 2024 stress.

This means **180 tested cells overall**, not ninety simultaneous tests.

Multiplicity correction is applied **separately within each 60-test family** using:

- Bonferroni family-wise error control;
- Benjamini-Hochberg false-discovery-rate control.

### Panel A stress status

The publication-resolution Panel A analysis found nominally significant cells, but **no cells survived Bonferroni or Benjamini-Hochberg correction** in any of the three 60-test stress families.

### Panel B stress status

**Not yet completed at publication resolution.**

The Panel B stress analysis is intentionally being held until the two-panel interpretation rules are frozen. Its purpose will be to determine whether the corrected stress conclusion is calendar-robust or calendar-sensitive.

No Panel B stress result should currently be inferred from the GOF analysis.

---

## Extreme-value diagnostics

POT/GPD analysis is applied to negative standardized residuals from the selected marginal reference models.

For Panel B, the q90 Cedi reference gives approximately:

```text
xi       = 0.693
beta     = 0.329
n_exc    = 278
VaR99    = 2.332 standardized-residual units
ES99     = 7.616 standardized-residual units
```

Across Cedi thresholds from `q = 0.85` to `0.975`, the estimated shape parameter remains positive, approximately `0.63` to `0.79`.

These values are treated as **conditional diagnostics**, not model-free structural facts, because no tested Cedi marginal passes the full adequacy gate.

Older EVT numbers in superseded manuscript drafts should not be treated as current publication evidence.

---

## Cedi robustness

The original business-day analysis includes seven Cedi treatments, labelled A-G, covering alternatives such as:

- baseline GARCH reference;
- raw ranks;
- AR-only filtering;
- weekly aggregation;
- hurdle treatment;
- static mixture treatment;
- Markov-switching treatment.

These remain valid as Panel A robustness evidence.

A complete seven-treatment rerun on Panel B is **not automatically required**. It will only be triggered if synchronized-panel stress results involving the Cedi materially change the substantive conclusion, or if a later review requires the full cross-panel matrix.

---

## Monthly transmission analysis

The previous monthly transmission results are **withdrawn from the current manuscript evidence**.

A data-integrity audit found invalid literal zero values in the extracted Bank of Ghana headline year-on-year inflation series for part of 2023. The prior monthly regression outputs therefore cannot be treated as publication findings.

The monthly transmission question will remain suspended until:

1. the CPI series is rebuilt from a verified authoritative source;
2. the monthly panel is re-audited;
3. the regression analysis is rerun from scratch.

The March 2025 zero monthly Cedi return was separately checked and is a genuine zero in the available Bank of Ghana series.

---

## Data sources

### Commodity prices

Daily-source-frequency commodity series:

- Cocoa: `CC=F`
- Gold: `GC=F`
- Brent crude oil: `BZ=F`
- WTI crude oil: `CL=F`

The repository records the provider symbols and retrieval workflow. The project does not assume that the provider's continuous-series construction eliminates futures-roll effects.

The World Bank Pink Sheet comparison is used only as a low-frequency external consistency check, not as proof that daily futures-roll effects are absent.

### Exchange rate

The canonical exchange-rate source is the **Bank of Ghana interbank USD/GHS mid-rate**.

The sign is reversed for dependence analysis so that:

```text
negative Cedi return = depreciation
```

This aligns adverse Cedi movements with the lower-tail orientation used for commodity-price declines.

The Yahoo `GHS=X` series is used only as a diagnostic comparator and is not the publication exchange-rate input.

---

## Repository structure

```text
paper/
  latex/                       Manuscript source and compiled paper artifacts
  VERIFICATION_LEDGER.csv      Claim-to-output verification ledger

src/tailrisk/
  marginals.py                 Marginal-model code
  evt.py                       POT/GPD and tail diagnostics
  copulas.py                   8-family bivariate copula estimation
  gof.py                       Copula goodness-of-fit machinery
  inference.py                 Bootstrap and marginal diagnostics
  networks.py                  Rolling/network analysis utilities

scripts/
  00-13                        Original data, analysis and robustness workflow
  23-27                        Data provenance and calendar-construction audits
  28                           Panel B marginal + EVT publication rebuild
  29                           Panel B 8-family GOF rebuild
  30                           Panel A hard convergence-gate audit
  31                           Planned Panel B publication stress comparison

data/
  raw/                         Source inputs permitted in the repository
  processed/                   Processed analysis inputs

outputs/
  tables/                      Original-panel and historical result tables
  figures/                     Figures
  original_panel_gate_audit/   Hard-gate audit for Panel A
  publication_rebuild_v2/      Panel B publication-rebuild outputs
```

Some older scripts and outputs remain for provenance and reproducibility. Their presence does not mean every historical claim remains active in the current manuscript.

---

## Current reproduction status

The repository is **not yet at a one-command final-publication reproduction stage**.

Do not assume that the legacy command:

```bash
python scripts/02_run_pipeline.py --data real
```

reproduces the current two-panel publication architecture. That script belongs to the earlier business-day workflow and is retained for provenance.

The current publication reconstruction proceeds through explicit audited stages:

### Panel A audit

```bash
python scripts/30_original_panel_convergence_gate_audit.py
```

This verifies the frozen 3,008-row input and applies the strengthened hard convergence gate without altering any existing analysis artifacts.

### Panel B marginal rebuild

```bash
python scripts/28_publication_rebuild_stage1.py
```

### Panel B copula GOF

```bash
python scripts/29_publication_rebuild_stage2_gof.py
```

### Panel B stress comparison

```bash
python scripts/31_publication_rebuild_stage3_stress.py
```

**Do not run Stage 31 merely because it is listed here.** It is the next planned expensive analysis and should only be run after the two-panel interpretation protocol is accepted and frozen.

A final top-level publication reproduction command will be added after the scientific architecture and manuscript are frozen.

---

## Reproducibility safeguards

The current publication workflow uses:

- SHA-256 hashes for frozen analysis inputs and outputs;
- deterministic seed derivation;
- optimizer-convergence gating;
- finite-value validity checks;
- explicit adequacy diagnostics;
- bootstrap checkpoints for long-running GOF jobs;
- separate output directories for historical and rebuilt analyses;
- non-destructive audit scripts;
- explicit separation between relative fit and absolute model adequacy.

The project does not silently replace historical outputs when a new audit is run.

---

## What is currently settled

The following points are considered established for the current reconstruction:

- the original 3,008-row Panel A is exactly identified and hash-frozen;
- the synchronized 2,774-row Panel B is exactly identified and hash-frozen;
- enforcing optimizer convergence does not change any selected Panel A marginal model;
- the Cedi remains marginally inadequate under both panel constructions;
- Gold-Brent, Gold-WTI and Brent-WTI static-copula inadequacy is robust across both panels;
- all four commodity-Cedi GOF non-rejections are robust across both panels;
- cocoa-related commodity GOF conclusions are calendar-sensitive;
- Panel A corrected stress inference finds no Bonferroni or BH discoveries;
- monthly transmission results are withdrawn pending CPI reconstruction.

---

## What remains open

The main remaining scientific tasks are:

1. freeze the two-panel interpretation protocol;
2. run the Panel B publication-resolution stress comparison;
3. classify stress conclusions as calendar-robust or calendar-sensitive;
4. decide whether any synchronized Cedi A-G follow-up is triggered;
5. rebuild the monthly macroeconomic panel if transmission remains in scope;
6. rewrite the manuscript from verified outputs;
7. refresh figures, tables and the verification ledger;
8. convert the final paper to the target journal format;
9. create a clean one-command reproducibility workflow for the frozen publication release.

---

## Manuscript status

The manuscript currently in `paper/latex/` contains historical material that is being reconstructed from the audited evidence base.

Until the rewrite is complete:

- do not treat every numerical claim in the current PDF as authoritative;
- prefer the audited CSV/JSON outputs and verification ledger;
- treat older manuscript drafts as historical artifacts.

---

## License

MIT. See `LICENSE`.
