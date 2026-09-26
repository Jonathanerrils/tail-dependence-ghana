# Publication figure set

This directory is the frozen manuscript figure set for the publication revision.

## Main manuscript
- `figure_01_calendar_workflow.png`
- `figure_02_tail_concentration_profiles.png`
- `figure_03_copula_gof_heatmap.png`
- `figure_04_stress_summary.png`

## Supplementary diagnostics
- `figure_S1_cedi_evt_threshold_sensitivity.png`
- `figure_S2_cedi_AG_sensitivity.png`
- `figure_S3_pit_clipping_diagnostic.png`

## Other publication assets
- `graphical_abstract_verified_findings.png`
- `cross_panel_evidence_summary.png` (visual reference; use a native LaTeX table in the manuscript)
- `repository_study_design_workflow.png`

## LaTeX usage

Main figures are referenced from `paper/latex/paper.tex` with paths such as:

```latex
\includegraphics[width=\linewidth]{figures/publication/figure_02_tail_concentration_profiles.png}
```

The manuscript should use the native LaTeX cross-panel evidence table rather than embedding the table-image asset.
