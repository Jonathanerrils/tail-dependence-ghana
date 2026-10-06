# International Journal of Finance & Economics submission package

Target journal: International Journal of Finance & Economics (Wiley)

Working branch: `ijfe-submission`

## Source of truth

The scientific source of truth remains the audited manuscript in `paper/latex/rebuild/` on the publication-revision history. This directory is a journal-specific condensed version. It preserves the frozen numerical findings and removes reconstruction detail that is not necessary for the main journal article.

## Current journal requirements reflected here

- Strict 8,000-word limit excluding references; title page, abstract and tables are excluded.
- Up to seven keywords.
- Main structure: Introduction; Materials/Methods; Results; Discussion; Conclusion.
- LaTeX is accepted for submission when accompanied by a compiled PDF and supporting files.
- Substantive generative-AI use is disclosed in the Methods section.
- Figures and tables are embedded for the initial review build; editable/separate files remain available in the package for revision.
- Data-sharing statement included.

## Free publication route

IJFE is free to submit. Open access is optional. The intended route for this manuscript is the standard subscription publication route, so no optional gold-open-access APC should be selected.

## Files

- `manuscript.tex` — main submission manuscript
- `supplementary.tex` — diagnostic supplementary material
- `references.bib` — bibliography
- `figures/` — frozen publication figures
- `sections/` — condensed article sections
- `cover_letter.tex` — journal cover-letter draft

## Scientific freeze

Do not alter the frozen sample sizes, copula GOF classifications, stress-test counts, Cedi marginal status or multiplicity-controlled conclusions without returning to the audited source and outputs.


## Data access and redistribution

The submission Data Availability Statement points directly to the `ijfe-submission` branch rather than the repository default branch. Raw commodity-price files obtained through Yahoo Finance are not redistributed because of provider terms; the repository documents the tickers and processing workflow needed to reconstruct the analysis from the original provider, subject to those terms. Bank of Ghana exchange-rate data are attributed to the original public source.
