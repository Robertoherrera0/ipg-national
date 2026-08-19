import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

adj = pd.read_csv("data/national/pruned/mutual/mutual_p0.35.csv", index_col=0)
adj.index   = adj.index.str.strip()
adj.columns = adj.columns.str.strip()
G = nx.from_pandas_adjacency(adj)
G.remove_edges_from(nx.selfloop_edges(G))

TARGET          = 0.85
damping_factors = np.round(np.arange(0.50, 0.96, 0.05), 2)

rank_matrix = {}
for d in damping_factors:
    pr = nx.pagerank(G, alpha=d, weight="weight")
    sorted_schools = sorted(pr, key=pr.get, reverse=True)
    rank_matrix[d] = {s: r+1 for r, s in enumerate(sorted_schools)}

schools = list(rank_matrix[TARGET].keys())
rank_df = pd.DataFrame(
    {d: [rank_matrix[d][s] for s in schools] for d in damping_factors},
    index=schools
)
schools = rank_df[TARGET].sort_values().index.tolist()
rank_df = rank_df.loc[schools]

n       = len(schools)
palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
colors  = {s: palette[i % len(palette)] for i, s in enumerate(schools)}

# spread ranks — multiply by spacing factor so names fit
SPACING = 3
rank_df_spaced = rank_df * SPACING

fig = go.Figure()

for school in schools:
    fig.add_trace(go.Scatter(
        x=damping_factors,
        y=rank_df_spaced.loc[school].values,
        mode="lines+markers",
        line=dict(color=colors[school], width=2),
        marker=dict(size=5),
        hovertemplate=f"<b>{school}</b><br>α=%{{x:.2f}}<br>Rank=%{{customdata}}<extra></extra>",
        customdata=rank_df.loc[school].values,
        showlegend=False,
    ))

# label only at α=0.85, on top of the dot
for school in schools:
    fig.add_annotation(
        x=TARGET,
        y=rank_df_spaced.loc[school, TARGET],
        text=school,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        yshift=6,
        font=dict(size=8, color=colors[school]),
    )

fig.add_vline(x=TARGET, line_width=1.5, line_dash="dash", line_color="#555")

fig.update_layout(
    title=dict(
        text="PageRank Rankings Remain Stable Across Damping Factors",
        x=0.5, font=dict(size=16)
    ),
    xaxis=dict(
        title="Damping Factor (α)",
        tickvals=list(damping_factors),
        ticktext=[str(d) for d in damping_factors],
        showgrid=False,
    ),
    yaxis=dict(
        title="Rank",
        autorange="reversed",
        tickvals=[i * SPACING for i in range(1, n + 1)],
        ticktext=list(range(1, n + 1)),
        showgrid=True,
        gridcolor="rgba(200,200,200,0.25)",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    width=1400,
    height=n * 28,   # auto height based on number of schools
    margin=dict(l=60, r=60, t=70, b=60),
)

fig.show()
fig.write_html("figures/fig_damping.html")
fig.write_image("figures/fig_damping.svg", width=1400, height=n*28)

