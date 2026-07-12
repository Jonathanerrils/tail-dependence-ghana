"""Build a single-file interactive dashboard (outputs/dashboard.html).

Panels: (1) calm vs stress lower-tail-dependence heatmaps,
(2) rolling 250-day co-crash dependence, (3) tail-event timeline —
days on which two or more series simultaneously breach their 5% loss
quantile.

Reads the canonical files written by scripts/02_run_pipeline.py, so it
works identically for --data synthetic and --data real (run the pipeline
first; for real data it adapts the ingestion script's output and writes
data/processed/prices_real.csv + returns_real.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"

ACCENT = "#7B4B2A"  # cocoa brown
FONT = "Georgia, 'Times New Roman', serif"


def heatmap_panel() -> go.Figure:
    calm = pd.read_csv(TAB / "lambda_matrix_calm.csv", index_col=0)
    s24 = pd.read_csv(TAB / "lambda_matrix_2024.csv", index_col=0)
    zmax = float(max(0.5, np.nanmax(calm.values), np.nanmax(s24.values)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Calm periods",
                                                        "2024 cocoa-shock window"))
    for i, m in enumerate([calm, s24], start=1):
        fig.add_trace(
            go.Heatmap(z=m.values, x=list(m.columns), y=list(m.index),
                       zmin=0, zmax=zmax, colorscale="YlOrRd",
                       showscale=(i == 2), text=np.round(m.values, 2),
                       texttemplate="%{text}",
                       colorbar=dict(title="λ<sub>L</sub>")),
            row=1, col=i)
    fig.update_layout(height=440, font=dict(family=FONT),
                      title="Lower-tail dependence λ̂<sub>L</sub>(q = 0.05)")
    fig.update_yaxes(autorange="reversed")
    return fig


def rolling_panel() -> go.Figure:
    roll = pd.read_csv(TAB / "rolling_lambda_L.csv", index_col=0, parse_dates=True)
    # prefer pairs involving cocoa or the cedi; fall back to the first six
    keep = [c for c in roll.columns
            if "cocoa" in c.lower() or "ghs" in c.lower()][:6] \
        or list(roll.columns)[:6]
    fig = go.Figure()
    for c in keep:
        fig.add_trace(go.Scatter(x=roll.index, y=roll[c], name=c,
                                 line=dict(width=1.4)))
    for a, b, lab in [("2020-03-01", "2020-06-30", "COVID"),
                      ("2024-01-01", "2024-12-31", "2024 cocoa shock")]:
        fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.06, line_width=0,
                      annotation_text=lab, annotation_position="top left")
    fig.update_layout(height=430, font=dict(family=FONT),
                      title="Rolling 250-day lower-tail dependence, "
                            "λ̂<sub>L</sub>(q = 0.10)",
                      yaxis_title="λ̂<sub>L</sub>",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def timeline_panel(data: str) -> go.Figure:
    rets = pd.read_csv(PROC / f"returns_{data}.csv", index_col=0, parse_dates=True)
    breach = rets.lt(rets.quantile(0.05))
    n_joint = breach.sum(axis=1)
    events = n_joint[n_joint >= 2]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=events.index, y=events.values, marker_color=ACCENT,
                         name="# series in joint 5% loss tail"))
    fig.update_layout(height=340, font=dict(family=FONT),
                      title="Tail-event timeline: joint 5%-quantile loss breaches",
                      yaxis_title="# series breaching")
    return fig


def provenance_line(data: str) -> str:
    if data != "real":
        return ("<p class='warn'>Demo run on <strong>synthetic</strong> data — "
                "run the ingestion script, then "
                "<code>scripts/02_run_pipeline.py --data real</code> and rebuild "
                "with <code>--data real</code> before interpreting results.</p>")
    rets = pd.read_csv(PROC / "returns_real.csv", index_col=0, parse_dates=True)
    return (f"<p class='warn'><strong>Real data.</strong> {rets.shape[1]} series "
            f"({', '.join(rets.columns)}), {rets.shape[0]} daily observations, "
            f"{rets.index.min().date()} → {rets.index.max().date()}. Sources: "
            f"Yahoo Finance futures &amp; GHS/USD (fetch-only licence — do not "
            f"redistribute raw data), FRED/EIA oil spot (public domain), World "
            f"Bank &amp; IMF macro. See the ingestion script's data dictionary "
            f"for details.</p>")


def main(data: str = "synthetic") -> None:
    panels = [heatmap_panel(), rolling_panel(), timeline_panel(data)]
    parts = [f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
             for i, f in enumerate(panels)]
    banner = provenance_line(data)
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Ghana Commodity Tail-Risk Dashboard</title>
<style>
 body {{ font-family: {FONT}; margin: 0; background: #FBF7F1; color: #26201A; }}
 header {{ border-bottom: 3px solid {ACCENT}; padding: 28px 40px 18px; }}
 h1 {{ margin: 0 0 4px; font-size: 26px; letter-spacing: .3px; }}
 header p {{ margin: 2px 0; color: #6B5B4D; font-size: 14px; }}
 .warn {{ background: #FDEBD2; border-left: 4px solid {ACCENT}; padding: 8px 12px;
          font-size: 13px; }}
 section {{ padding: 10px 40px 24px; }}
 footer {{ padding: 14px 40px 30px; font-size: 12px; color: #8A7A6B; }}
</style></head><body>
<header>
 <h1>Tail Dependence &amp; Extreme Commodity Risk — Ghana Exposure</h1>
 <p>Cocoa · Gold · Brent · WTI · GHS/USD — copula-EVT tail-risk monitor</p>
 {banner}
</header>
<section>{parts[0]}</section>
<section>{parts[1]}</section>
<section>{parts[2]}</section>
<footer>Built by scripts/03_build_dashboard.py · empirical λ<sub>L</sub> on
GARCH-filtered PIT residuals · stress windows: COVID (Mar–Jun 2020), 2024
cocoa shock.</footer>
</body></html>"""
    out = ROOT / "outputs" / "dashboard.html"
    out.write_text(html)
    print(f"Dashboard written to {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    main(ap.parse_args().data)
