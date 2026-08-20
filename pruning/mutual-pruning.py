import os
import pandas as pd
import networkx as nx

raw_matrix_csv = "data/national/national_ipg_matrix.csv"

output_dir = "data/national/pruned/mutual"
os.makedirs(output_dir, exist_ok=True)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def compute_top_partners(adj, percent):
    top_partners = {}

    for school in adj.index:
        weights = adj.loc[school].drop(school)
        weights = weights[weights > 0]
        weights = weights.sort_values(ascending=False)

        k = max(1, int(len(weights) * percent))
        top_partners[school] = set(weights.head(k).index)

    return top_partners


def prune_graph(adj, percent):
    top_partners = compute_top_partners(adj, percent)

    G = nx.Graph()
    G.add_nodes_from(adj.index)

    for i in adj.index:
        for j in top_partners[i]:
            if i in top_partners.get(j, set()):
                G.add_edge(i, j)

    return G


def graph_to_adj_df(G, nodes):
    df = pd.DataFrame(0, index=nodes, columns=nodes)
    for u, v in G.edges():
        df.loc[u, v] = 1
        df.loc[v, u] = 1
    return df


def main():
    adj = pd.read_csv(raw_matrix_csv, index_col=0)

    for p in thresholds:
        G = prune_graph(adj, p)
        pruned_adj = graph_to_adj_df(G, adj.index)

        filename = f"mutual_p{p}.csv"
        output_path = os.path.join(output_dir, filename)
        pruned_adj.to_csv(output_path)

        print(f"Saved: {filename} | Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()