# LaTeX manuscript rebuild

This directory contains the reconstructed manuscript assembled from the audited scientific drafts.

## Build

From `paper/latex/rebuild`:

```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

or, if available:

```bash
latexmk -pdf manuscript.tex
```

## Structure

- `manuscript.tex`: master file
- `references.bib`: bibliography used by the rebuild
- `sections/01_introduction.tex` through `sections/08_conclusion.tex`: manuscript sections

## Scientific status

This rebuild follows the frozen two-panel interpretation:
- Panel A and Panel B are co-equal calendar constructions.
- Stress conclusions are based on separately corrected 60-test families.
- Gold--Brent, Gold--WTI and Brent--WTI show calendar-robust 8/8 static-copula rejection.
- Cocoa--Gold, Cocoa--Brent and Cocoa--WTI are calendar-sensitive in GOF.
- Commodity--Cedi GOF results are non-rejections in both panels, but Cedi inference remains conditional on marginal inadequacy.
- The earlier monthly transmission branch is excluded because the macroeconomic input series failed audit.

The existing `paper/latex/paper.tex` is left untouched as historical material. This rebuild is intentionally isolated in `paper/latex/rebuild/` until it passes consistency, bibliography and compilation checks.
