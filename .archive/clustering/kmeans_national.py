import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

excel_file = "data/national/network_metrics.xlsx"

sheets = [
    "national_ipg_matrix",
    "national_ipg_matrix_relative",
    "national_ipg_matrix_unweighted"
]

output_dir = "analysis/pca/pca_outputs"
os.makedirs(output_dir, exist_ok=True)


def analyze_sheet(sheet_name):

    print(f"Processing sheet: {sheet_name}")

    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    schools = df.index
    metrics = df.copy()

    scaler = StandardScaler()
    Z = scaler.fit_transform(metrics)

    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics.corr(), cmap="coolwarm", center=0)
    plt.title(f"Correlation Matrix - {sheet_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sheet_name}_correlation.png")
    plt.close()

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    explained = pca.explained_variance_ratio_

    print("PCA Variance Explained:")
    print(f"PC1: {explained[0]*100:.2f}%")
    print(f"PC2: {explained[1]*100:.2f}%")

    kmeans = KMeans(n_clusters=2, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(Z)

    plt.figure(figsize=(10, 8))

    colors = ["red", "blue"]

    for cluster_id in np.unique(clusters):
        subset = Z_pca[clusters == cluster_id]
        subset_schools = schools[clusters == cluster_id]

        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            color=colors[cluster_id],
            label=f"Cluster {cluster_id}",
            s=70
        )

        for i, school in enumerate(subset_schools):
            plt.text(
                subset[i, 0],
                subset[i, 1],
                school,
                fontsize=7
            )

    plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    plt.title(f"PCA Scatter (k=2) - {sheet_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sheet_name}_pca_k2.png")
    plt.close()

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=metrics.columns
    )

    print("PC1 Loadings:")
    print(loadings["PC1"].sort_values(ascending=False))

    plt.figure(figsize=(8, 5))
    loadings["PC1"].sort_values().plot(kind="barh")
    plt.title(f"PC1 Loadings - {sheet_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sheet_name}_pc1_loadings.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    loadings["PC2"].sort_values().plot(kind="barh")
    plt.title(f"PC2 Loadings - {sheet_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sheet_name}_pc2_loadings.png")
    plt.close()


for sheet in sheets:
    analyze_sheet(sheet)
