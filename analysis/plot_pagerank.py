import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

corr = pd.read_csv("pagerank_correlations.csv", index_col=0)

values = corr.iloc[:, 0]
corr_matrix = pd.DataFrame(values)
corr_matrix.columns = ["PageRank"]

fig, ax = plt.subplots(figsize=(6,8))

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

ax.set_title("Spearman Correlation with PageRank", fontsize=14, pad=12)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)

plt.tight_layout()
plt.savefig("pagerank_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
