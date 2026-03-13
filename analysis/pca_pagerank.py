import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("data/ipg_metrics_with_pagerank.csv")

internal_metrics = [
    "EdgesPerNode",
    "assortativity_unweighted",
    "global_efficiency",
    "local_efficiency",
    "avg_path_over_diameter",
    "clique_integration",
    "sigma_small_world_index",
    "omega_small_world_index"
]

X = df[internal_metrics]

# -----------------------------
# Standardize
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA
# -----------------------------

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

df["PC1"] = components[:,0]
df["PC2"] = components[:,1]

print("\nExplained variance:")
print(pca.explained_variance_ratio_)

# -----------------------------
# Regression: PageRank ~ PCA
# -----------------------------

X_reg = sm.add_constant(df[["PC1","PC2"]])
y = df["pagerank"]

model = sm.OLS(y, X_reg).fit()

print("\nRegression: PageRank ~ Structural Components\n")
print(model.summary())

# -----------------------------
# KMeans clustering
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(df["cluster"].value_counts())

# -----------------------------
# Plot: PCA structural map
# -----------------------------

sns.set_style("whitegrid")

plt.figure(figsize=(10,8))

sns.scatterplot(
    data=df,
    x="PC1",
    y="PC2",
    hue="cluster",
    size="pagerank",
    sizes=(50,400),
    palette="Set2"
)

plt.title("Institution Structural Space (PCA)")
plt.xlabel("Structural Component 1")
plt.ylabel("Structural Component 2")

plt.tight_layout()

plt.savefig("figures/institution_structure_map.png", dpi=600)

plt.show()

# -----------------------------
# Plot: clusters vs PageRank
# -----------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="cluster",
    y="pagerank",
    palette="Set2"
)

plt.title("PageRank by Institutional Collaboration Type")
plt.xlabel("Institution Cluster")
plt.ylabel("PageRank")

plt.tight_layout()

plt.savefig("figures/pagerank_by_cluster.png", dpi=600)

plt.show()