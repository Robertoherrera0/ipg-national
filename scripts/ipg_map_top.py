import pandas as pd
import numpy as np
import networkx as nx

# ---------------- CONFIG ----------------
matrix_csv = "../data/national/national_ipg_matrix.csv"
output_csv = "../data/national/national_ipg_mutual_20.csv"
threshold_percent = 0.20
min_weight = 1
# ----------------------------------------


def compute_top_partners(adj, top_percent):
    top_partners = {}
    for school in adj.index:
        weights = adj.loc[school].drop(school)
        weights = weights[weights >= min_weight]
        weights = weights.sort_values(ascending=False)
        k = max(1, int(len(weights) * top_percent))
        top_partners[school] = set(weights.head(k).index)
    return top_partners


def build_mutual_graph(adj, top_partners):
    G = nx.Graph()
    G.add_nodes_from(adj.index)

    for i in adj.index:
        for j in top_partners[i]:
            if i in top_partners[j]:
                G.add_edge(i, j, weight=adj.loc[i, j])

    return G


def main():
    adj = pd.read_csv(matrix_csv, index_col=0)

    top_partners = compute_top_partners(adj, threshold_percent)
    G = build_mutual_graph(adj, top_partners)

    mutual_adj = nx.to_pandas_adjacency(G, weight="weight")
    mutual_adj = mutual_adj.reindex(index=adj.index, columns=adj.index, fill_value=0)
    mutual_adj.to_csv(output_csv)

    print(f"Saved mutual {int(threshold_percent*100)}% graph to {output_csv}")

    components = list(nx.connected_components(G))

    component_map = {}
    for comp_id, comp in enumerate(components):
        for node in comp:
            component_map[node] = comp_id + 1  # start at 1

    for node in adj.index:
        if node not in component_map:
            component_map[node] = None

    component_df = pd.DataFrame({
        "School": adj.index,
        "Component": [component_map[node] for node in adj.index]
    })

    component_output = output_csv.replace(".csv", "_components.csv")
    component_df.to_csv(component_output, index=False)

    print(f"Saved component membership to {component_output}")

if __name__ == "__main__":
    main()