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
