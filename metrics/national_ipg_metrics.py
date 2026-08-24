import os
import pandas as pd
import networkx as nx

NATIONAL_FILE = "data/national/pruned/mutual/mutual_p0.35.csv"
OUTPUT_FILE = "data/national/school_external_metrics.csv"


def main():
    print("Loading pruned national network...")
    national_adj = pd.read_csv(NATIONAL_FILE, index_col=0)
    G = nx.from_pandas_adjacency(national_adj)

    print(f"Network loaded: {G.number_of_nodes()} schools, {G.number_of_edges()} edges\n")
    print("Computing external (national) network metrics (weighted)...")

    # For distance-based measures, invert weight (higher collaboration = shorter distance)
    G_dist = G.copy()
    for u, v, d in G_dist.edges(data=True):
        d["distance"] = 1.0 / d["weight"] if d["weight"] > 0 else float("inf")

    results = pd.DataFrame(index=G.nodes())

    results["pagerank"] = pd.Series(
        nx.pagerank(G, alpha=0.85, weight="weight")
    )
    results["degree_centrality"] = pd.Series(
        nx.degree_centrality(G)  # unweighted by definition (normalized count of neighbors)
    )
    results["weighted_degree"] = pd.Series(
        dict(G.degree(weight="weight"))
    )
    results["eigenvector_centrality"] = pd.Series(
        nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    )
    results["closeness_centrality"] = pd.Series(
        nx.closeness_centrality(G_dist, distance="distance")
    )
    results["betweenness_centrality"] = pd.Series(
        nx.betweenness_centrality(G_dist, weight="distance")
    )
    results["clustering"] = pd.Series(
        nx.clustering(G, weight="weight")
    )

    results = results.reset_index().rename(columns={"index": "school"})

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved external metrics to {OUTPUT_FILE}")
    print(f"{len(results)} schools x {len(results.columns)} columns")
    print(f"Columns: {list(results.columns)}")


if __name__ == "__main__":
    main()