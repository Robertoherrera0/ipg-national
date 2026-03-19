import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/ipg_metrics_with_pagerank.csv")

cols = [
    "pagerank",
    "EdgesPerNode",
    "clique_integration",
    "global_efficiency",
    "local_efficiency"
]

corr_matrix = df[cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    cbar=True,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size":11},
    ax=ax
)

ax.set_title("Spearman Correlation Matrix (Internal Metrics)", fontsize=14, pad=12)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)

plt.tight_layout()
plt.show()

cols = [
    "pagerank",
    "degree_centrality",
    "eigenvector_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "clustering"
]

corr_matrix = df[cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    cbar=True,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size":11},
    ax=ax
)

ax.set_title("Spearman Correlation (External Metrics)", fontsize=14, pad=12)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)

plt.tight_layout()
plt.show()