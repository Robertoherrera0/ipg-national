"""
step1_build_metrics.py
─────────────────────────────────────────────────────────────────────
Computes internal metrics for all schools + national centrality.
Saves: data/ipg_combined_metrics_v2.csv

Internal (per school adjacency):
  - EdgesPerNode
  - clique_integration  (largest 3-clique community / n nodes)
  - global_efficiency
  - local_efficiency

National (mutual_p0.35.csv):
  - pagerank, degree_centrality, eigenvector_centrality,
    closeness_centrality, betweenness_centrality, clustering

Run: python step1_build_metrics.py
─────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import k_clique_communities

SCHOOLS_DIR   = "data/schools"
NATIONAL_FILE = "data/national/pruned/mutual/mutual_p0.35.csv"
OUT_CSV       = "data/ipg_combined_metrics_v2.csv"

# ── INTERNAL METRIC FUNCTIONS ─────────────────────────────────────

def compute_edges_per_node(G):
    return G.number_of_edges() / G.number_of_nodes()

def compute_clique_integration(G, k=3):
    """Largest 3-clique community / total nodes"""
    try:
        communities = list(k_clique_communities(G, k))
        if not communities:
            return 0.0
        return max(len(c) for c in communities) / G.number_of_nodes()
    except:
        return np.nan

def compute_internal_metrics(adj_path):
    df = pd.read_csv(adj_path, index_col=0)
    G  = nx.from_pandas_adjacency(df)
    return {
        "EdgesPerNode":       compute_edges_per_node(G),
        "clique_integration": compute_clique_integration(G),
        "global_efficiency":  nx.global_efficiency(G),
        "local_efficiency":   nx.local_efficiency(G),
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
        metrics           = compute_internal_metrics(graph_path)
        metrics["school"] = school
        nan_cols = [k for k, v in metrics.items()
                    if isinstance(v, float) and np.isnan(v)]
        flag = f"  [NaN: {nan_cols}]" if nan_cols else ""
        print(f"  ✓ {school}{flag}")
        internal_results.append(metrics)
    except Exception as e:
        print(f"  ✗ {school}  ERROR: {e}")

internal_df = pd.DataFrame(internal_results)
print(f"\n  → {len(internal_df)} schools processed")

# ── NATIONAL NETWORK ──────────────────────────────────────────────
print("\nComputing national network centrality...")

national_adj = pd.read_csv(NATIONAL_FILE, index_col=0)
Gnat         = nx.from_pandas_adjacency(national_adj)

national_df  = pd.DataFrame(index=Gnat.nodes())
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
final_df = pd.merge(internal_df, national_df, on="school")
cols     = ["school"] + [c for c in final_df.columns if c != "school"]
final_df = final_df[cols]

os.makedirs("data", exist_ok=True)
final_df.to_csv(OUT_CSV, index=False)

print(f"\n  Saved: {OUT_CSV}")
print(f"  {len(final_df)} rows  ×  {len(final_df.columns)} columns")
print(f"  Columns: {list(final_df.columns)}")
print("\nRun step2_analysis.py next.")