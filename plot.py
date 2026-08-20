import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_excel("IPG_metrics_updated.xlsx")
df = df.replace([np.inf,-np.inf],np.nan).dropna()

metrics = [
    "EdgesPerNode",
    "assortativity_unweighted",
    "assortativity_weighted",
    "avg_path_over_diameter",
    "clique_integration",
    "global_efficiency",
    "local_efficiency",
    "sigma_small_world_index",
    "omega_small_world_index"
]

df = df.dropna(subset=metrics)

remove = ["maine"]

df_fit = df[~df["School"].str.lower().isin(remove)].copy()
df_out = df[df["School"].str.lower().isin(remove)].copy()

scaler = StandardScaler()
Z_fit = scaler.fit_transform(df_fit[metrics])
Z_out = scaler.transform(df_out[metrics]) if len(df_out) > 0 else None

kmeans = KMeans(n_clusters=3, n_init=50, random_state=42)
labels = kmeans.fit_predict(Z_fit)

pca = PCA(n_components=2)
pc_fit = pca.fit_transform(Z_fit)
pc_out = pca.transform(Z_out) if Z_out is not None else None

loadings = pd.DataFrame(
    pca.components_.T,
    index=metrics,
    columns=["PC1","PC2"]
)

colors = {0:"green",1:"orange",2:"blue"}
cluster_colors = pd.Series(labels).map(colors)

fig, axes = plt.subplots(1, 2, figsize=(22,8))

axes[0].scatter(pc_fit[:,0], pc_fit[:,1], c=cluster_colors, s=90, edgecolor="black")

for i, name in enumerate(df_fit["School"]):
    axes[0].text(pc_fit[i,0], pc_fit[i,1], name, fontsize=7)

if pc_out is not None:
    axes[0].scatter(pc_out[:,0], pc_out[:,1], c="grey", s=110, edgecolor="black")
    for i, name in enumerate(df_out["School"]):
        axes[0].text(pc_out[i,0], pc_out[i,1], name, fontsize=8, color="grey")

axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].grid(True)

loadings.plot(kind="barh", ax=axes[1])
axes[1].grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
