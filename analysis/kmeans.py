import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C_DARK  = "#1A3A5C"
C_LIGHT = "#5B9EC9"

schools = {
    "MU":       ("data/schools/MU/graphs/MU_adjacency.csv", "high"),
    "PennState": ("data/schools/PennState/graphs/PennState_adjacency.csv", "low")
}

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["", ""])

for col, (school, (path, tier)) in enumerate(schools.items(), 1):
    adj = pd.read_csv(path, index_col=0)
    G = nx.from_pandas_adjacency(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    pos = nx.spring_layout(G, seed=42, k=2)
    degrees = dict(G.degree())

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_color = "rgba(26,58,92,0.3)" if tier == "high" else "rgba(91,158,201,0.6)"

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color=edge_color),
        hoverinfo="none",
        showlegend=False
    ), row=1, col=col)

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_sizes = [12 + min(degrees[n], 8) * 2 for n in G.nodes()]
    node_color = C_DARK if tier == "high" else C_LIGHT

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=node_sizes, color=node_color,
                    opacity = 0.9,
                    line=dict(width=0.5, color="white")),
        hoverinfo="none",
        showlegend=False
    ), row=1, col=col)

fig.update_layout(
    title=dict(text=""),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    width=1200,
    height=600,
    margin=dict(l=40, r=40, t=20, b=20),
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

fig.write_image("figures/network_comparison.svg")
fig.write_html("figures/network_comparison.html")
fig.show()