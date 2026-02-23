import os
import glob
import pandas as pd
import numpy as np
import networkx as nx

schools_root = "data/schools"
output_file = "data/national/school_internal_metrics.csv"


def load_adjacency(path):
    return pd.read_csv(path, index_col=0)


def compute_metrics(adj):

    G = nx.from_pandas_adjacency(adj)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    total_internal_weight = adj.values.sum() / 2

    density = nx.density(G)

    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))

    avg_degree = np.mean(list(degrees.values())) if n_nodes > 0 else 0
    avg_weighted_degree = np.mean(list(weighted_degrees.values())) if n_nodes > 0 else 0

    clustering = nx.average_clustering(G)

    if n_nodes > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_component_ratio = len(largest_cc) / n_nodes
    else:
        largest_component_ratio = 0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "total_internal_weight": total_internal_weight,
        "density": density,
        "avg_degree": avg_degree,
        "avg_weighted_degree": avg_weighted_degree,
        "avg_clustering": clustering,
        "largest_component_ratio": largest_component_ratio
    }


def main():

    records = []

    school_dirs = glob.glob(os.path.join(schools_root, "*"))

    for school_path in school_dirs:

        school_name = os.path.basename(school_path)

        graph_path = os.path.join(
            school_path,
            "graphs",
            f"{school_name}_adjacency.csv"
        )

        if not os.path.exists(graph_path):
            continue

        adj = load_adjacency(graph_path)

        metrics = compute_metrics(adj)
        metrics["school"] = school_name

        records.append(metrics)

        print(f"Processed {school_name}")

    df = pd.DataFrame(records)
    df = df.set_index("school")
    df.to_csv(output_file)

    print(f"\nSaved metrics to {output_file}")


if __name__ == "__main__":
    main()