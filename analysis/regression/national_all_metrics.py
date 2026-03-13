import os
import pandas as pd
import numpy as np
import networkx as nx

schools_dir = "data/schools"
national_file = "data/national/pruned/mutual/mutual_p0.35.csv"

N_RANDOM = 20
MAX_TRIES = 50


def largest_component(G):
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def compute_edges_per_node(G):
    return G.number_of_edges() / G.number_of_nodes()


from networkx.algorithms.community import k_clique_communities

def clique_integration(G, k=3):

    try:
        communities = list(k_clique_communities(G, k))

        if len(communities) == 0:
            return 0

        largest = max(len(c) for c in communities)

        return largest / G.number_of_nodes()

    except:
        return np.nan

def avg_path_over_diameter(G):
    Gc = largest_component(G)
    try:
        avg = nx.average_shortest_path_length(Gc)
        diam = nx.diameter(Gc)
        if diam == 0:
            return np.nan
        return avg / diam
    except:
        return np.nan


def generate_connected_random_graph(n, m):
    for _ in range(MAX_TRIES):
        G = nx.gnm_random_graph(n, m)
        if nx.is_connected(G):
            return G
    return None


def small_world_sigma(G):
    Gc = largest_component(G)

    try:
        n = Gc.number_of_nodes()
        m = Gc.number_of_edges()

        C = nx.average_clustering(Gc)
        L = nx.average_shortest_path_length(Gc)

        Cr_vals = []
        Lr_vals = []

        for _ in range(N_RANDOM):
            rand = generate_connected_random_graph(n, m)
            if rand is None:
                continue
            Cr_vals.append(nx.average_clustering(rand))
            Lr_vals.append(nx.average_shortest_path_length(rand))

        if len(Cr_vals) == 0:
            return np.nan

        Cr = np.mean(Cr_vals)
        Lr = np.mean(Lr_vals)

        if Cr == 0 or Lr == 0:
            return np.nan

        return (C / Cr) / (L / Lr)

    except:
        return np.nan


def small_world_omega(G):
    Gc = largest_component(G)

    try:
        n = Gc.number_of_nodes()

        C = nx.average_clustering(Gc)
        L = nx.average_shortest_path_length(Gc)

        lattice = nx.watts_strogatz_graph(n, 4, 0)
        lattice = largest_component(lattice)

        Cl = nx.average_clustering(lattice)
        Ll = nx.average_shortest_path_length(lattice)

        rand_L_vals = []

        for _ in range(N_RANDOM):
            rand = generate_connected_random_graph(n, Gc.number_of_edges())
            if rand is None:
                continue
            rand_L_vals.append(nx.average_shortest_path_length(rand))

        if len(rand_L_vals) == 0:
            return np.nan

        Lr = np.mean(rand_L_vals)

        if Cl == 0:
            return np.nan

        return (Lr / L) - (C / Cl)

    except:
        return np.nan


def compute_internal_metrics(adj_path):
    df = pd.read_csv(adj_path, index_col=0)
    G = nx.from_pandas_adjacency(df)

    metrics = {}

    metrics["EdgesPerNode"] = compute_edges_per_node(G)

    try:
        metrics["assortativity_unweighted"] = nx.degree_assortativity_coefficient(G)
    except:
        metrics["assortativity_unweighted"] = np.nan

    metrics["global_efficiency"] = nx.global_efficiency(G)
    metrics["local_efficiency"] = nx.local_efficiency(G)
    metrics["avg_path_over_diameter"] = avg_path_over_diameter(G)
    metrics["clique_integration"] = clique_integration(G)
    metrics["sigma_small_world_index"] = small_world_sigma(G)
    metrics["omega_small_world_index"] = small_world_omega(G)

    return metrics


internal_results = []

for school in os.listdir(schools_dir):

    graph_path = os.path.join(
        schools_dir,
        school,
        "graphs",
        f"{school}_adjacency.csv"
    )

    if not os.path.exists(graph_path):
        continue

    metrics = compute_internal_metrics(graph_path)
    metrics["school"] = school

    if any(pd.isna(v) for v in metrics.values()):
        print(school)

    internal_results.append(metrics)

internal_df = pd.DataFrame(internal_results)


national_adj = pd.read_csv(national_file, index_col=0)
Gnat = nx.from_pandas_adjacency(national_adj)

national_df = pd.DataFrame(index=Gnat.nodes())

national_df["degree_centrality"] = pd.Series(nx.degree_centrality(Gnat))
national_df["eigenvector_centrality"] = pd.Series(nx.eigenvector_centrality(Gnat))
national_df["closeness_centrality"] = pd.Series(nx.closeness_centrality(Gnat))
national_df["betweenness_centrality"] = pd.Series(nx.betweenness_centrality(Gnat))
national_df["clustering"] = pd.Series(nx.clustering(Gnat))

national_df = national_df.reset_index().rename(columns={"index": "school"})


final_df = pd.merge(internal_df, national_df, on="school")

cols = ["school"] + [c for c in final_df.columns if c != "school"]
final_df = final_df[cols]

final_df.to_csv("data/ipg_combined_metrics.csv", index=False)

print("Saved to data/ipg_combined_metrics.csv")