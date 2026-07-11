"""Tail-risk networks.

Nodes are assets/macro variables; edge weights are lower-tail dependence
estimates. Provides rolling-window co-crash dynamics and calm-vs-stress
network snapshots.
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .copulas import empirical_lambda_lower


def rolling_lower_tail(
    pit: pd.DataFrame, window: int = 250, q: float = 0.10
) -> dict[tuple[str, str], pd.Series]:
    """Rolling empirical lower-tail dependence for every column pair.

    Ranks are recomputed within each window so the estimator adapts to
    local marginal behaviour.
    """
    out: dict[tuple[str, str], pd.Series] = {}
    cols = list(pit.columns)
    idx = pit.index[window - 1:]
    for a, b in itertools.combinations(cols, 2):
        vals = np.full(len(idx), np.nan)
        xa, xb = pit[a].to_numpy(), pit[b].to_numpy()
        for i in range(window - 1, len(pit)):
            ua = stats.rankdata(xa[i - window + 1 : i + 1]) / (window + 1)
            ub = stats.rankdata(xb[i - window + 1 : i + 1]) / (window + 1)
            vals[i - window + 1] = empirical_lambda_lower(ua, ub, q=q)
        out[(a, b)] = pd.Series(vals, index=idx, name=f"{a}–{b}")
    return out


def tail_dependence_matrix(pit: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    cols = list(pit.columns)
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        lam = empirical_lambda_lower(pit[a].to_numpy(), pit[b].to_numpy(), q=q)
        m.loc[a, b] = m.loc[b, a] = lam
    return m


def build_network(lam: pd.DataFrame, min_weight: float = 0.02) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(lam.columns)
    for a, b in itertools.combinations(lam.columns, 2):
        w = float(lam.loc[a, b])
        if w >= min_weight:
            g.add_edge(a, b, weight=w)
    return g


def plot_networks(
    lam_calm: pd.DataFrame,
    lam_stress: pd.DataFrame,
    titles: tuple[str, str],
    path: str,
    node_colors: dict[str, str] | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    pos = nx.circular_layout(sorted(lam_calm.columns))
    vmax = max(lam_calm.to_numpy().max(), lam_stress.to_numpy().max())
    for ax, lam, title in zip(axes, [lam_calm, lam_stress], titles):
        g = build_network(lam)
        weights = [g[u][v]["weight"] for u, v in g.edges()]
        colors = (
            [node_colors.get(n, "#4477AA") for n in g.nodes()]
            if node_colors
            else "#4477AA"
        )
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2400, node_color=colors,
                               edgecolors="white", linewidths=2)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_color="white",
                                font_weight="bold")
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            width=[10 * w for w in weights],
            edge_color=weights, edge_cmap=plt.cm.Reds,
            edge_vmin=0, edge_vmax=vmax, alpha=0.9,
        )
        for (u, v), w in zip(g.edges(), weights):
            mid = (pos[u] + pos[v]) / 2
            ax.text(*mid, f"{w:.2f}", fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("Lower-tail dependence networks (edge = empirical $\\lambda_L$)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
