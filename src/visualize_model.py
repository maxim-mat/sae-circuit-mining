"""
Minimal notebook visualizers for Part 2 output.

Reads the artifacts written by part2_ordering.run():
    dfg_<cf>.edges.parquet   columns: src, dst, weight
    dfg_<cf>.nodes.parquet   columns: activity, start, end
    net_<cf>.pnml            (if pm4py discovery succeeded)

Usage in a notebook
-------------------
    from viz_part2 import load_dfg, view_dfg, view_dfg_mpl, view_net

    dfg, starts, ends = load_dfg("nets/dfg_s2_io_flip")
    view_dfg(dfg, starts, ends)          # pm4py/Graphviz, rich
    view_dfg_mpl(dfg, starts, ends)      # matplotlib fallback, no Graphviz
    view_net("nets/net_s2_io_flip.pnml") # the discovered Petri net
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Load the exported DFG parquets back into pm4py-shaped objects
# ---------------------------------------------------------------------------

def load_dfg(path_prefix: str):
    """Read dfg_<cf>.{edges,nodes}.parquet -> (dfg, starts, ends).
    path_prefix is the part WITHOUT the .edges/.nodes suffix, e.g.
    'nets/dfg_s2_io_flip'."""
    e = pd.read_parquet(path_prefix + ".edges.parquet")
    n = pd.read_parquet(path_prefix + ".nodes.parquet")
    dfg = {(r.src, r.dst): float(r.weight) for r in e.itertuples(index=False)}
    starts = {r.activity: float(r.start) for r in n.itertuples(index=False) if r.start > 0}
    ends = {r.activity: float(r.end) for r in n.itertuples(index=False) if r.end > 0}
    return dfg, starts, ends


# ---------------------------------------------------------------------------
# DFG views
# ---------------------------------------------------------------------------

def view_dfg(dfg: dict, starts: dict, ends: dict, fmt: str = "png"):
    """Rich DFG view via pm4py (needs Graphviz). Renders inline in a notebook."""
    import pm4py
    pm4py.view_dfg(dfg, dict(starts), dict(ends), format=fmt)


def view_dfg_mpl(dfg: dict, starts: dict, ends: dict, figsize=(9, 7),
                 layer_from_label=True):
    """Graphviz-free DFG view with matplotlib + networkx. Nodes are laid out
    by layer if activity labels look like 'L03.F11' (Part 2's convention),
    else by a spring layout. Edge width ~ weight; start/end nodes highlighted.
    Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    for (a, b), w in dfg.items():
        g.add_edge(a, b, weight=w)

    if layer_from_label and all("." in n for n in g.nodes()):
        # x = layer, y = spread within layer
        layers = {}
        for n in g.nodes():
            L = int(n.split(".")[0][1:])
            layers.setdefault(L, []).append(n)
        pos = {}
        for L, nodes in layers.items():
            for i, n in enumerate(sorted(nodes)):
                pos[n] = (L, i - (len(nodes) - 1) / 2)
    else:
        pos = nx.spring_layout(g, seed=0)

    weights = [g[u][v]["weight"] for u, v in g.edges()]
    wmax = max(weights) if weights else 1.0
    widths = [0.5 + 3.5 * w / wmax for w in weights]

    node_colors = []
    for n in g.nodes():
        if n in starts and n in ends:
            node_colors.append("#c8a2c8")
        elif n in starts:
            node_colors.append("#8ecae6")
        elif n in ends:
            node_colors.append("#ffb703")
        else:
            node_colors.append("#e0e0e0")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(g, pos, width=widths, edge_color="#888",
                           arrowsize=12, connectionstyle="arc3,rad=0.08", ax=ax)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=900,
                           edgecolors="#444", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    ax.set_xlabel("layer" if layer_from_label else "")
    ax.set_title("DFG (blue=start, orange=end, purple=both; width ~ weight)")
    ax.axis("off" if not layer_from_label else "on")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Petri net view
# ---------------------------------------------------------------------------

def view_net(pnml_path: str, fmt: str = "png"):
    """Read a saved PNML and render the Petri net via pm4py (needs Graphviz)."""
    import pm4py
    net, im, fm = pm4py.read_pnml(pnml_path)
    pm4py.view_petri_net(net, im, fm, format=fmt)
    return net, im, fm


def save_net_image(pnml_path: str, out_path: str):
    """Render a saved PNML to an image file (png/svg by extension)."""
    import pm4py
    net, im, fm = pm4py.read_pnml(pnml_path)
    pm4py.save_vis_petri_net(net, im, fm, out_path)
    return out_path
