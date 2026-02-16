import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("regression_dataset.csv").replace([np.inf, -np.inf], np.nan).dropna()


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

ridge_alpha = 1.0
lasso_alpha = 0.01

folder = "regression_by_cluster_fixed_alpha"
os.makedirs(folder, exist_ok=True)

#cluster loop
cluster_labels = ["Cluster 1", "Cluster 2", "Cluster 3"]

for cluster_label in cluster_labels:
    df_cluster = df[df["Group"] == cluster_label].reset_index(drop=True)
    n_schools = len(df_cluster)
    print(f"Processing {cluster_label} with {n_schools} schools...")
    

    X = df_cluster[metrics].values
    y = df_cluster["log_avg_papers"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ols = LinearRegression().fit(X_scaled, y)
    ridge = Ridge(alpha=ridge_alpha).fit(X_scaled, y)
    lasso = Lasso(alpha=lasso_alpha, max_iter=200000).fit(X_scaled, y)
    
    y_pred_ols = ols.predict(X_scaled)
    y_pred_ridge = ridge.predict(X_scaled)
    y_pred_lasso = lasso.predict(X_scaled)
    
    r2_ols = ols.score(X_scaled, y)
    r2_ridge = ridge.score(X_scaled, y)
    r2_lasso = lasso.score(X_scaled, y)

    mse_ols = np.mean((y - y_pred_ols)**2)
    mse_ridge = np.mean((y - y_pred_ridge)**2)
    mse_lasso = np.mean((y - y_pred_lasso)**2)
    
    results = pd.DataFrame({
        "metric": metrics,
        "ols_coef": ols.coef_,
        "ridge_coef": ridge.coef_,
        "lasso_coef": lasso.coef_
    })

    summary = pd.DataFrame({
        "model": ["OLS", "Ridge", "Lasso"],
        "R2": [r2_ols, r2_ridge, r2_lasso],
        "MSE": [mse_ols, mse_ridge, mse_lasso],
        "alpha": [np.nan, ridge_alpha, lasso_alpha]
    })

    
    plt.figure(figsize=(18, 5))
    
    for i, (model_name, y_pred, r2, mse, alpha) in enumerate([
        ("OLS", y_pred_ols, r2_ols, mse_ols, np.nan),
        ("Ridge", y_pred_ridge, r2_ridge, mse_ridge, ridge_alpha),
        ("Lasso", y_pred_lasso, r2_lasso, mse_lasso, lasso_alpha)
    ]):
        plt.subplot(1, 3, i+1)
        plt.scatter(y, y_pred, edgecolor='black')
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        plt.plot(lims, lims, 'r--')
        title = f"{model_name}"
        if not np.isnan(alpha):
            title += f" (alpha={alpha})"
        title += f"\nR²={r2:.3f}, MSE={mse:.3f}"
        plt.title(title)
        plt.xlabel("Actual")
        if i == 0:
            plt.ylabel("Predicted")
    
    plt.tight_layout()
    plt.savefig(f"{folder}/{cluster_label.replace(' ', '_')}_actual_vs_predicted.png")
    plt.close()
    

    plt.figure(figsize=(12, 6))
    results.set_index("metric")[["ols_coef", "ridge_coef", "lasso_coef"]].plot(kind="bar")
    plt.title(f"{cluster_label} Model Coefficients")
    plt.tight_layout()
    plt.savefig(f"{folder}/{cluster_label.replace(' ', '_')}_coefficients_barplot.png")
    plt.close()


    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(X_scaled, columns=metrics).corr(), annot=True, cmap="coolwarm", center=0)
    plt.title(f"{cluster_label} Predictor Correlation")
    plt.tight_layout()
    plt.savefig(f"{folder}/{cluster_label.replace(' ', '_')}_predictor_correlation_heatmap.png")
    plt.close()
    

    excel_path = f"{folder}/{cluster_label.replace(' ', '_')}_regression_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        results.to_excel(writer, sheet_name="Coefficients", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    
    print(f"{cluster_label} done! Results saved to {excel_path}\n")


