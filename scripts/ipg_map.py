# This builds a geographic collaboration network between IPG schools.
# mode = "single":
#     For each school, we show only its top x% most collaborative schools.
#
# mode = "pair":
#     We only show an edge if both schools are in each other's top x%.
#
# Node size reflects collaborations within each school's IPG.
# Edge thickness reflects number of joint publications between the two schools.

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

top_percent = 0.05
min_weight = 1
mode = "single"

# controls how much thicker strong collaborations appear compared to weak ones
edge_width_scale = 8
# minimum edge line thickness
edge_width_min = 0.8

# controls how much larger highly collaborative schools appear from internal IPG collabs
node_size_scale = 22
# minimum node size
node_size_min = 5

matrix_csv = "data/national/national_ipg_matrix.csv"
internal_csv = "data/national/school_internal_metrics.csv"
coord_csv = "data/national/school_coordinates.csv"

def load_coordinates(path):
    df = pd.read_csv(path)
    return {
        row["school"]: (row["latitude"], row["longitude"])
        for _, row in df.iterrows()
    }

def load_matrix(path):
    return pd.read_csv(path, index_col=0)


def load_internal_metrics(path):
    return pd.read_csv(path, index_col=0)


def compute_top_partners(adj):
    top_partners = {}
    for school in adj.index:
        weights = adj.loc[school].drop(school)
        weights = weights[weights >= min_weight]
        weights = weights.sort_values(ascending=False)
        k = max(1, int(len(weights) * top_percent))
        top_partners[school] = set(weights.head(k).index)
    return top_partners


def build_graph(adj, top_partners):
    G = nx.Graph()
    for school in adj.index:
        G.add_node(school)

    if mode == "single":
        for i in adj.index:
            for j in top_partners[i]:
                G.add_edge(i, j, weight=adj.loc[i, j])
        title = f"Top {int(top_percent*100)}% Collaborations Per School"
    else:
        for i in adj.index:
            for j in top_partners[i]:
                if i in top_partners[j]:
                    G.add_edge(i, j, weight=adj.loc[i, j])
        title = f"Top {int(top_percent*100)}% Mutual Collaborations"

    return G, title


def create_plot(G, adj, internal_df, title):
    fig = go.Figure()
    coords = load_coordinates(coord_csv)

    edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if len(edge_weights) == 0:
        edge_weights = [1]

    min_w, max_w = min(edge_weights), max(edge_weights)
    if max_w == min_w:
        max_w = min_w + 1

    for u, v, d in G.edges(data=True):
        lat1, lon1 = coords[u]
        lat2, lon2 = coords[v]
        w = d["weight"]

        width = edge_width_min + edge_width_scale * ((w - min_w) / (max_w - min_w))

        fig.add_trace(
            go.Scattergeo(
                lon=[lon1, lon2],
                lat=[lat1, lat2],
                mode="lines",
                line=dict(width=width, color="rgba(30,30,120,0.45)"),
                hoverinfo="text",
                text=f"{u} ↔ {v}<br>Joint publications: {w}"
            )
        )

    internal = internal_df["total_internal_weight"]

    total_publications_across_all_schools = adj.copy()
    np.fill_diagonal(total_publications_across_all_schools.values, 0)
    total_publications_across_all_schools = total_publications_across_all_schools.sum(axis=1)

    min_i, max_i = internal.min(), internal.max()
    if max_i == min_i:
        max_i = min_i + 1

    node_lats = []
    node_lons = []
    node_sizes = []
    node_labels = []
    node_text = []

    for node in G.nodes:

        lat, lon = coords[node]
        internal_value = internal[node]

        size = node_size_min + node_size_scale * ((internal_value - min_i) / (max_i - min_i))

        neighbors = list(G.neighbors(node))
        n_top_partners = len(neighbors)

        total_weight_to_top = sum(adj.loc[node, nbr] for nbr in neighbors)

        node_lats.append(lat)
        node_lons.append(lon)
        node_sizes.append(size)
        node_labels.append(node)

        node_text.append(
            f"<b>{node}</b><br>"
            f"Top partner schools shown: {n_top_partners}<br>"
            f"Publications with top partners: {total_weight_to_top}<br>"
            f"Total publications across all schools: {int(total_publications_across_all_schools[node])}<br>"
            f"Internal publications (within school IPG): {int(internal_value)}<br>"
            f"IPG members: {internal_df.loc[node,'n_nodes']}"
        )

    fig.add_trace(
        go.Scattergeo(
            lon=node_lons,
            lat=node_lats,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            marker=dict(
                size=node_sizes,
                color="crimson",
                opacity=0.95,
                line=dict(width=1.5, color="black")
            ),
            hoverinfo="text",
            hovertext=node_text
        )
    )

    fig.update_layout(
        title=f"IPG Collaboration Network — {title}",
        showlegend=False,
        geo=dict(scope="usa", projection_type="albers usa"),
        width=1500,
        height=900
    )

    return fig


def main():

    adj = load_matrix(matrix_csv)
    internal_df = load_internal_metrics(internal_csv)

    top_partners = compute_top_partners(adj)
    G, title = build_graph(adj, top_partners)
    fig = create_plot(G, adj, internal_df, title)

    filename = f"ipg_network_{mode}_{int(top_percent*100)}.html"
    fig.write_html(filename)
    fig.show()


if __name__ == "__main__":
    main()