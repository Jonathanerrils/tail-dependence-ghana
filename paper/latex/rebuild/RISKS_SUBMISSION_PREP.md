# Risks submission-format preparation

Target journal: *Risks* (MDPI)

This file records the remaining presentation and metadata work needed to migrate the audited manuscript into the current MDPI LaTeX template.

## Current publication assets

Main manuscript figures:
1. Calendar-construction workflow
2. Empirical finite-threshold tail-concentration profiles
3. Static copula goodness-of-fit heatmap
4. Stress--calm nominal-versus-adjusted evidence

Supplementary figures:
- Figure S1: Panel B Cedi EVT threshold sensitivity
- Figure S2: Panel A Cedi A--G sensitivity diagnostics
- Figure S3: Panel A Cedi PIT-clipping diagnostic

Other submission assets:
- Graphical abstract
- Repository-only study-design workflow

## Presentation audit completed

- Main figures are legible and appear close to their first discussion.
- Cross-panel evidence is a native LaTeX table rather than a raster table image.
- Supplementary figures are separated onto individual pages for legibility.
- Table captions and labels are normalized.
- Figure and table cross-references have been checked.
- The abstract is 192 words and fits the usual MDPI expectation of about 200 words.
- The manuscript uses 8 keywords, within MDPI's recommended range.
- The historical study-design workflow is not counted as a main manuscript figure.

## Metadata still required from the author before the MDPI template can be finalized

Do not invent these fields:
- Full institutional affiliation, including city and country
- Corresponding-author email
- ORCID, if available
- Funding statement
- Conflict-of-interest statement
- Acknowledgments, if any

## Back matter required for the final MDPI version

The final *Risks* manuscript should include:
- Supplementary Materials
- Author Contributions
- Funding
- Data Availability Statement
- Acknowledgments (if applicable)
- Conflicts of Interest
- References

The existing repository Data and Code Availability statement can be adapted into the MDPI Data Availability Statement.

## Template migration

Use the current MDPI LaTeX template for *Risks*. The audited scientific content in `paper/latex/rebuild/sections/` should be migrated without changing numerical results or evidential classifications. References should be rendered in MDPI's numbered citation style through the journal template rather than the present author--year `plainnat` development style.

The final submission ZIP should contain the MDPI source, bibliography, all main figures, the graphical abstract where requested, and any supplementary files needed for recompilation.
