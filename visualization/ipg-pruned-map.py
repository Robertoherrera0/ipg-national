import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px


adjacency_path = "data/national/pruned/threshold/threshold_p0.1_f10.csv"
coord_csv = "data/national/school_coordinates.csv"


def load_coordinates(path):
    df = pd.read_csv(path)
    return {
        row["school"]: (row["latitude"], row["longitude"])
        for _, row in df.iterrows()
    }


def create_component_map(G, coords, save_path, title):

    fig = go.Figure()

    components = list(nx.connected_components(G))
    component_map = {}
    for idx, comp in enumerate(components):
        for node in comp:
            component_map[node] = idx

    colors = px.colors.qualitative.Set1
    n_colors = len(colors)

    for u, v in G.edges():
        lat1, lon1 = coords[u]
        lat2, lon2 = coords[v]

        fig.add_trace(
            go.Scattergeo(
                lon=[lon1, lon2],
                lat=[lat1, lat2],
                mode="lines",
                line=dict(width=1.5, color="rgba(30,30,120,0.4)"),
                hoverinfo="text",
                text=f"{u} ↔ {v}",
                showlegend=False
            )
        )

    for comp_id in set(component_map.values()):
        nodes_in_comp = [n for n in G.nodes if component_map[n] == comp_id]

        lats = [coords[n][0] for n in nodes_in_comp]
        lons = [coords[n][1] for n in nodes_in_comp]

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="markers+text",
                text=nodes_in_comp,
                textposition="top center",
                marker=dict(
                    size=10,
                    color=colors[comp_id % n_colors],
                    line=dict(width=1, color="black")
                ),
                name=f"Component {comp_id+1}"
            )
        )

    fig.update_layout(
        title=title,
        geo=dict(scope="usa", projection_type="albers usa"),
        showlegend=True,
        width=1500,
        height=900
    )

    fig.write_html(save_path)
    fig.show()


def main():

    adj = pd.read_csv(adjacency_path, index_col=0)
    coords = load_coordinates(coord_csv)

    G = nx.from_pandas_adjacency(adj)

    # Remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Determine output folder
    if "mutual" in adjacency_path.lower():
        save_dir = "visualization/pruned/mutual"
    else:
        save_dir = "visualization/pruned/threshold"

    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(adjacency_path).replace(".csv", ".html")
    save_path = os.path.join(save_dir, filename)

    title = f"Pruned Network: {filename}"

    create_component_map(G, coords, save_path, title)


if __name__ == "__main__":
    main()