import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_FILE = "data/school_internal_metrics.csv"
OUTPUT_DIR = "results/tables"

METRICS = [
    "EdgesPerNode",
    "assortativity_unweighted",
    "assortativity_weighted",
    "avg_path_over_diameter",
    "clique_integration",
    "global_efficiency",
    "local_efficiency",
]


def load_data():
    df = pd.read_csv(DATA_FILE)
    before = len(df)
    df = df.dropna(subset=METRICS)
    after = len(df)
    print(f"Loaded {before} schools, {after} retained after dropping missing values "
          f"in: {METRICS}")
    return df


def run_pca(df):
    X = df[METRICS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    return pca, scores


def print_variance_explained(pca):
    print("\n" + "=" * 70)
    print("EXPLAINED VARIANCE")
    print("=" * 70)
    print(f"\n{'Component':<12s} {'Var. Explained':>15s} {'Cumulative':>15s}")
    print("-" * 45)
    
    rows = []
    cum = 0
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        cum += var
        rows.append({
            "Component": f"PC{i}",
            "Variance_Explained": var,
            "Cumulative": cum
        })
        print(f"PC{i:<10d} {var:>14.3f} {cum:>15.3f}")
    
    var_df = pd.DataFrame(rows)
    var_df.to_csv(os.path.join(OUTPUT_DIR, "pca_variance_explained.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/pca_variance_explained.csv")


def print_loadings(pca, n_components=3):
    print("\n" + "=" * 70)
    print(f"LOADINGS (first {n_components} components)")
    print("=" * 70)
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=METRICS,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    print(f"\n{loadings.round(3)}")
    
    loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))
    print(f"Saved: {OUTPUT_DIR}/pca_loadings.csv")


def print_scores(df, scores, n_components=3):
    print("\n" + "=" * 70)
    print(f"SCHOOL SCORES (first {n_components} components)")
    print("=" * 70)
    score_df = pd.DataFrame(
        scores[:, :n_components],
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    school_col = "School" if "School" in df.columns else "school"
    score_df.insert(0, "School", df[school_col].values)
    score_df = score_df.sort_values("PC1", ascending=False)
    print(f"\n{score_df.round(3).to_string(index=False)}")
    
    score_df.to_csv(os.path.join(OUTPUT_DIR, "pca_school_scores.csv"), index=False)
    print(f"Saved: {OUTPUT_DIR}/pca_school_scores.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    pca, scores = run_pca(df)

    print_variance_explained(pca)
    print_loadings(pca)
    print_scores(df, scores)


if __name__ == "__main__":
    main()