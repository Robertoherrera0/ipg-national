"""
step1_build_all_metrics.py
─────────────────────────────────────────────────────────────────────
Computes ALL internal metrics (8) + ALL external metrics (6) including PageRank.
Saves: data/ipg_all_metrics.csv

Internal (per school adjacency) — 8 metrics:
  1. EdgesPerNode
  2. assortativity
  3. global_efficiency
  4. local_efficiency
  5. avg_path_over_diameter
  6. clique_integration
  7. sigma (small-world index)
  8. omega (small-world index)

External (national network) — 6 metrics:
  1. pagerank
  2. degree_centrality
  3. eigenvector_centrality
  4. closeness_centrality
  5. betweenness_centrality
  6. clustering

Run: python step1_build_all_metrics.py
─────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import k_clique_communities

SCHOOLS_DIR   = "data/schools"
NATIONAL_FILE = "data/national/pruned/mutual/mutual_p0.35.csv"
OUT_CSV       = "data/ipg_all_metrics.csv"

N_RANDOM  = 20
MAX_TRIES = 50

# ── HELPER FUNCTIONS ──────────────────────────────────────────────

def largest_component(G):
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

def generate_connected_random_graph(n, m):
    for _ in range(MAX_TRIES):
        G = nx.gnm_random_graph(n, m)
        if nx.is_connected(G):
            return G
    return None

# ── INTERNAL METRIC FUNCTIONS ─────────────────────────────────────

def compute_edges_per_node(G):
    return G.number_of_edges() / G.number_of_nodes()

def compute_assortativity(G):
    try:
        return nx.degree_assortativity_coefficient(G)
    except:
        return np.nan

def compute_clique_integration(G, k=3):
    """Size of largest k-clique community / total nodes"""
    try:
        communities = list(k_clique_communities(G, k))
        if not communities:
            return 0.0
        largest_community_size = max(len(c) for c in communities)
        return largest_community_size / G.number_of_nodes()
    except:
        return np.nan

def compute_avg_path_over_diameter(G):
    Gc = largest_component(G)
    try:
        avg = nx.average_shortest_path_length(Gc)
        diam = nx.diameter(Gc)
        if diam == 0:
            return np.nan
        return avg / diam
    except:
        return np.nan

def compute_sigma(G):
    """Small-world sigma index"""
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

def compute_omega(G):
    """Small-world omega index"""
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
    """Compute all 8 internal metrics"""
    df = pd.read_csv(adj_path, index_col=0)
    G  = nx.from_pandas_adjacency(df)
    
    return {
        "EdgesPerNode":             compute_edges_per_node(G),
        "assortativity":            compute_assortativity(G),
        "global_efficiency":        nx.global_efficiency(G),
        "local_efficiency":         nx.local_efficiency(G),
        "avg_path_over_diameter":   compute_avg_path_over_diameter(G),
        "clique_integration":       compute_clique_integration(G),
        "sigma":                    compute_sigma(G),
        "omega":                    compute_omega(G),
    }

# ── LOOP ALL SCHOOLS ──────────────────────────────────────────────
print("Computing internal metrics for all schools...\n")
internal_results = []

for school in sorted(os.listdir(SCHOOLS_DIR)):
    graph_path = os.path.join(SCHOOLS_DIR, school, "graphs",
                              f"{school}_adjacency.csv")
    if not os.path.exists(graph_path):
        continue
    
    try:
        metrics = compute_internal_metrics(graph_path)
        metrics["school"] = school
        
        # Check for NaN values
        nan_cols = [k for k, v in metrics.items()
                    if isinstance(v, float) and np.isnan(v)]
        flag = f"  [NaN: {', '.join(nan_cols)}]" if nan_cols else ""
        
        print(f"  ✓ {school}{flag}")
        internal_results.append(metrics)
        
    except Exception as e:
        print(f"  ✗ {school}  ERROR: {e}")

internal_df = pd.DataFrame(internal_results)
print(f"\n  → {len(internal_df)} schools processed")

# ── NATIONAL NETWORK ──────────────────────────────────────────────
print("\nComputing national network centrality (6 metrics)...")

national_adj = pd.read_csv(NATIONAL_FILE, index_col=0)
Gnat         = nx.from_pandas_adjacency(national_adj)

national_df = pd.DataFrame(index=Gnat.nodes())
national_df["pagerank"]               = pd.Series(
    nx.pagerank(Gnat, alpha=0.85, weight="weight"))
national_df["degree_centrality"]      = pd.Series(nx.degree_centrality(Gnat))
national_df["eigenvector_centrality"] = pd.Series(nx.eigenvector_centrality(Gnat))
national_df["closeness_centrality"]   = pd.Series(nx.closeness_centrality(Gnat))
national_df["betweenness_centrality"] = pd.Series(nx.betweenness_centrality(Gnat))
national_df["clustering"]             = pd.Series(nx.clustering(Gnat))

national_df = national_df.reset_index().rename(columns={"index": "school"})
print(f"  → {len(national_df)} schools in national network")

# ── MERGE & SAVE ──────────────────────────────────────────────────
final_df = pd.merge(internal_df, national_df, on="school", how="inner")
cols     = ["school"] + [c for c in final_df.columns if c != "school"]
final_df = final_df[cols]

os.makedirs("data", exist_ok=True)
final_df.to_csv(OUT_CSV, index=False)

print(f"\n{'='*70}")
print(f"  ✓ SAVED: {OUT_CSV}")
print(f"{'='*70}")
print(f"  Schools:  {len(final_df)}")
print(f"  Columns:  {len(final_df.columns)}")
print(f"\n  INTERNAL METRICS (8):")
internal_cols = ["EdgesPerNode", "assortativity", "global_efficiency",
                 "local_efficiency", "avg_path_over_diameter",
                 "clique_integration", "sigma", "omega"]
for i, col in enumerate(internal_cols, 1):
    print(f"    {i}. {col}")

print(f"\n  EXTERNAL METRICS (6):")
external_cols = ["pagerank", "degree_centrality", "eigenvector_centrality",
                 "closeness_centrality", "betweenness_centrality", "clustering"]
for i, col in enumerate(external_cols, 1):
    print(f"    {i}. {col}")

print(f"\n{'='*70}")
print("Run step2_complete_analysis.py next.")