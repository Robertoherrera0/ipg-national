import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_excel("IPG_metrics_updated.xlsx")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

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

kmeans = KMeans(n_clusters=3, n_init=50, random_state=42)
df_fit["Cluster"] = kmeans.fit_predict(Z_fit)

flat_rows = []
for _, r in df.iterrows():
    s = r["School"]
    if s.lower() == "maine":
        flat_rows.append({"School": s, "Group": "Outlier"})
    else:
        grp = int(df_fit.loc[df_fit["School"] == s, "Cluster"].iloc[0])
        flat_rows.append({"School": s, "Group": f"Cluster {grp+1}"})

flat_df = pd.DataFrame(flat_rows)

g1 = df_fit[df_fit["Cluster"] == 0].merge(df, on="School")
g2 = df_fit[df_fit["Cluster"] == 1].merge(df, on="School")
g3 = df_fit[df_fit["Cluster"] == 2].merge(df, on="School")
out_full = df_out.copy()

with pd.ExcelWriter("IPG_groups.xlsx", engine="openpyxl") as writer:
    flat_df.to_excel(writer, sheet_name="Flat", index=False)
    g1.to_excel(writer, sheet_name="Cluster1", index=False)
    g2.to_excel(writer, sheet_name="Cluster2", index=False)
    g3.to_excel(writer, sheet_name="Cluster3", index=False)
    out_full.to_excel(writer, sheet_name="Outliers", index=False)

print("Saved IPG_groups.xlsx")
