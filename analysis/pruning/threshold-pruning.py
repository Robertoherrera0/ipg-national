import os
import pandas as pd
import networkx as nx

raw_matrix_csv = "data/national/national_ipg_matrix.csv"
relative_matrix_csv = "data/national/national_ipg_matrix_relative.csv"

output_dir = "data/national/pruned/threshold"
os.makedirs(output_dir, exist_ok=True)

# Smaller, realistic grids
thresholds = [0.03, 0.05, 0.07, 0.10, 0.12]
floors = [10, 20, 30, 40, 50]


def prune_graph(rel_adj, raw_adj, percent_threshold, floor):
    G = nx.Graph()
    G.add_nodes_from(rel_adj.index)

    for i in rel_adj.index:
        for j in rel_adj.columns:
            if i == j:
                continue

            if rel_adj.loc[i, j] >= percent_threshold:
                if raw_adj.loc[i, j] >= floor:
                    G.add_edge(i, j)

    return G


def graph_to_adj_df(G, nodes):
    df = pd.DataFrame(0, index=nodes, columns=nodes)
    for u, v in G.edges():
        df.loc[u, v] = 1
        df.loc[v, u] = 1
    return df


def main():
    raw_adj = pd.read_csv(raw_matrix_csv, index_col=0)
    rel_adj = pd.read_csv(relative_matrix_csv, index_col=0)

    for p in thresholds:
        for f in floors:
            G = prune_graph(rel_adj, raw_adj, p, f)
            pruned_adj = graph_to_adj_df(G, raw_adj.index)

            filename = f"threshold_p{p}_f{f}.csv"
            output_path = os.path.join(output_dir, filename)
            pruned_adj.to_csv(output_path)

            print(f"Saved: {filename} | Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()