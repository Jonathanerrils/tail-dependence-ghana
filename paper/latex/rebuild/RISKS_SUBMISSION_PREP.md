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

## Author metadata supplied

- Author: Jonathan Anthonio Nii Pedro Nelson
- Affiliation: Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana
- Corresponding-author email: jonathannelson707@gmail.com
- ORCID: 0009-0006-9393-6237
- Funding: no external funding
- Conflicts of Interest: none declared
- Acknowledgments: supplied and inserted into the development manuscript

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

Use the current MDPI LaTeX template for *Risks*. The audited scientific content in `paper/latex/rebuild/sections/` should be migrated without changing numerical results or evidential classifications. The current MDPI class identifies *Risks* as a Chicago-style journal, so the submission package uses the journal's author--date bibliography configuration rather than the development `plainnat` style.

The final submission ZIP should contain the MDPI source, bibliography, all main figures, the graphical abstract where requested, and any supplementary files needed for recompilation.


## Generative-AI disclosure check

Current MDPI author guidance asks authors to disclose generative-AI use when such tools were used in manuscript preparation or related research tasks. Because ChatGPT has been used during the reconstruction and manuscript-preparation workflow, the final MDPI submission should include an appropriate disclosure. The exact wording should be confirmed by the author before the final submission package is frozen.
