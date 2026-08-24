import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr

DATA_FILE = "data/national/pruned/mutual/mutual_p0.35.csv"
OUTPUT_DIR = "results/tables"

df = pd.read_csv(DATA_FILE, index_col=0)
df = df.loc[df.index, df.index]

G = nx.from_pandas_adjacency(df)

alphas = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

pr_scores = {}
pr_ranks = {}

for a in alphas:
    pr = nx.pagerank(G, alpha=a, weight="weight")
    pr_scores[a] = pr
    pr_ranks[a] = pd.Series(pr).rank(ascending=False, method="min")

scores_df = pd.DataFrame(pr_scores)
ranks_df = pd.DataFrame(pr_ranks)

os.makedirs(OUTPUT_DIR, exist_ok=True)

scores_df.to_csv(os.path.join(OUTPUT_DIR, "pagerank_scores_by_alpha.csv"))
print(f"Saved: {OUTPUT_DIR}/pagerank_scores_by_alpha.csv")

ranks_df.to_csv(os.path.join(OUTPUT_DIR, "pagerank_ranks_by_alpha.csv"))
print(f"Saved: {OUTPUT_DIR}/pagerank_ranks_by_alpha.csv")

national_df = pd.DataFrame(index=G.nodes())

national_df["pagerank"] = pd.Series(pr_scores[0.85])

national_df["degree_centrality"] = pd.Series(nx.degree_centrality(G))
national_df["eigenvector_centrality"] = pd.Series(nx.eigenvector_centrality(G))
national_df["closeness_centrality"] = pd.Series(nx.closeness_centrality(G))
national_df["betweenness_centrality"] = pd.Series(nx.betweenness_centrality(G))
national_df["clustering"] = pd.Series(nx.clustering(G))

metrics = [
    "degree_centrality",
    "eigenvector_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "clustering"
]

corr = {}

for m in metrics:
    tmp = national_df[[m, "pagerank"]].dropna()
    corr[m] = spearmanr(tmp[m], tmp["pagerank"]).correlation

corr_series = pd.Series(corr)
corr_series.to_csv(os.path.join(OUTPUT_DIR, "pagerank_correlations.csv"))
print(f"Saved: {OUTPUT_DIR}/pagerank_correlations.csv")

stab = pd.DataFrame(index=alphas, columns=alphas)

def align_corr(d1, d2):
    common = list(set(d1.keys()).intersection(set(d2.keys())))
    v1 = np.array([d1[n] for n in common])
    v2 = np.array([d2[n] for n in common])
    return spearmanr(v1, v2).correlation

for a in alphas:
    for b in alphas:
        stab.loc[a, b] = align_corr(pr_scores[a], pr_scores[b])

stab.to_csv(os.path.join(OUTPUT_DIR, "pagerank_stability.csv"))
print(f"Saved: {OUTPUT_DIR}/pagerank_stability.csv")