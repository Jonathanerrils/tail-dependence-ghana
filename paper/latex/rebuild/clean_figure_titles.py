from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

# Crop only the obsolete embedded title band.  The scientific content, axes,
# legends, annotations and footnotes are left pixel-for-pixel unchanged.
CROPS = {
    "figure_01_calendar_workflow.png": (941, 145),
    "repository_study_design_workflow.png": (941, 175),
    "figure_02_tail_concentration_profiles.png": (1392, 45),
    "figure_03_copula_gof_heatmap.png": (1995, 65),
    "figure_04_stress_summary.png": (1245, 60),
    "cross_panel_evidence_summary.png": (941, 90),
    "figure_S1_cedi_evt_threshold_sensitivity.png": (729, 55),
    "figure_S2_cedi_AG_sensitivity.png": (806, 55),
    "figure_S3_pit_clipping_diagnostic.png": (845, 55),
    "graphical_abstract_verified_findings.png": (941, 70),
}

for name, (original_height, crop_top) in CROPS.items():
    path = FIG / name
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        if im.height == original_height:
            cleaned = im.crop((0, crop_top, im.width, im.height))
            cleaned.save(path, optimize=True)
            print(f"cleaned {name}: {im.size} -> {cleaned.size}")
        elif im.height == original_height - crop_top:
            print(f"already clean: {name}")
        else:
            raise RuntimeError(
                f"Unexpected height for {name}: {im.height}; "
                f"expected {original_height} or {original_height - crop_top}"
            )


# Once title bands are physically removed from the PNGs, remove LaTeX-side
# trim values so captions/axes are not cropped a second time.
TEXT_REPLACEMENTS = {
    ROOT / "sections" / "04_methodology.tex": [
        (r"\\includegraphics[width=\\linewidth,trim=0 0 0 180,clip]{figures/repository_study_design_workflow.png}",
         r"\\includegraphics[width=\\linewidth]{figures/repository_study_design_workflow.png}"),
    ],
    ROOT / "sections" / "05_results.tex": [
        (r"\\includegraphics[width=\\linewidth,trim=0 0 0 48,clip]{figures/figure_02_tail_concentration_profiles.png}",
         r"\\includegraphics[width=\\linewidth]{figures/figure_02_tail_concentration_profiles.png}"),
        (r"\\includegraphics[width=0.90\\linewidth,trim=0 0 0 65,clip]{figures/figure_03_copula_gof_heatmap.png}",
         r"\\includegraphics[width=0.90\\linewidth]{figures/figure_03_copula_gof_heatmap.png}"),
        (r"\\includegraphics[width=\\linewidth,trim=0 0 0 58,clip]{figures/figure_04_stress_summary.png}",
         r"\\includegraphics[width=\\linewidth]{figures/figure_04_stress_summary.png}"),
        (r"\\includegraphics[width=0.96\\linewidth,trim=0 0 0 110,clip]{figures/cross_panel_evidence_summary.png}",
         r"\\includegraphics[width=0.96\\linewidth]{figures/cross_panel_evidence_summary.png}"),
    ],
    ROOT / "sections" / "09_supplementary_figures.tex": [
        (r"\\includegraphics[width=0.90\\linewidth,trim=0 0 0 70,clip]{figures/figure_S1_cedi_evt_threshold_sensitivity.png}",
         r"\\includegraphics[width=0.90\\linewidth]{figures/figure_S1_cedi_evt_threshold_sensitivity.png}"),
        (r"\\includegraphics[width=0.90\\linewidth,trim=0 0 0 70,clip]{figures/figure_S2_cedi_AG_sensitivity.png}",
         r"\\includegraphics[width=0.90\\linewidth]{figures/figure_S2_cedi_AG_sensitivity.png}"),
        (r"\\includegraphics[width=0.90\\linewidth,trim=0 0 0 70,clip]{figures/figure_S3_pit_clipping_diagnostic.png}",
         r"\\includegraphics[width=0.90\\linewidth]{figures/figure_S3_pit_clipping_diagnostic.png}"),
    ],
    ROOT / "graphical_abstract.tex": [
        (r"\\includegraphics[width=\\textwidth,trim=0 0 0 55,clip]{figures/graphical_abstract_verified_findings.png}",
         r"\\includegraphics[width=\\textwidth]{figures/graphical_abstract_verified_findings.png}"),
    ],
}

for path, replacements in TEXT_REPLACEMENTS.items():
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"updated {path.relative_to(ROOT)}")
