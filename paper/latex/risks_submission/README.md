# Risks submission package

This directory is the journal-format working package for submission to *Risks* (MDPI).

## Source of scientific content

The scientific text and numerical results are copied from the audited master in `paper/latex/rebuild/`. The audited master remains the source of truth. Journal-format changes in this directory must not silently change scientific results or evidential classifications.

## Main files

- `manuscript.tex` — Risks/MDPI-formatted main manuscript
- `supplementary.tex` — separate supplementary diagnostic figures
- `references.bib` — bibliography copied from the audited master
- `figures/` — frozen main and supplementary publication figures
- `graphical_abstract.png` — graphical abstract submission asset
- `Definitions/` — MDPI-compatible class and bibliography style files used for repository compilation

## Citation style

The Risks option in the MDPI class uses the journal's Chicago-style author-date bibliography configuration.

## Author metadata

- Jonathan Anthonio Nii Pedro Nelson
- Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana
- Corresponding email: jonathannelson707@gmail.com
- ORCID: 0009-0006-9393-6237
- Funding: no external funding
- Conflicts of interest: none declared

## Template note

The bundled MDPI class is used to validate journal-compatible layout in GitHub. The submit-mode MDPI logo call has been replaced with text so the repository build does not depend on a binary logo asset. Before final journal upload, compare the package against the official MDPI LaTeX template distributed by the journal and replace the Definitions files with the current official versions if required.

## Scientific freeze

Do not modify reported sample sizes, stress counts, GOF classifications, Cedi diagnostic status, or cross-panel conclusions without returning to the audited master and frozen outputs.
