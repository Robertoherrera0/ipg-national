import pandas as pd
import numpy as np
import os

root = os.getcwd()

metrics = pd.read_excel("IPG_metrics_updated.xlsx")

original_cols = [
    "School",
    "Nodes",
    "Edges",
    "EdgesPerNode",
    "assortativity_unweighted",
    "assortativity_weighted",
    "average_shortest_path_length",
    "diameter",
    "avg_path_over_diameter",
    "k",
    "n_cliques",
    "largest_clique_size",
    "overlap_nodes",
    "clique_integration",
    "global_efficiency",
    "local_efficiency",
    "sigma_small_world_index",
    "omega_small_world_index"
]

metrics = metrics[original_cols].copy()

groups = pd.read_excel("IPG_groups.xlsx", sheet_name="Flat")
groups = groups[groups["Group"].str.contains("Cluster", na=False)]

groups = groups[groups["School"].str.lower() != "maine"]

df = metrics.merge(groups, on="School", how="inner")

rows = []

for _, row in df.iterrows():

    school = row["School"]
    stats_path = os.path.join(root, school, "graphs", f"{school}_stats.csv")

    if not os.path.exists(stats_path):
        print(f"Missing stats for {school}")
        continue

    s = pd.read_csv(stats_path)

    faculty_count = s.shape[0]
    total_papers = s["Total papers"].sum()
    avg_papers = total_papers / faculty_count if faculty_count > 0 else np.nan
    log_avg = np.log(avg_papers) if avg_papers > 0 else np.nan
    collab_ratio = s["Collaborative papers"].sum() / s["Total papers"].sum()
    solo_ratio = s["Solo papers"].sum() / s["Total papers"].sum()

    avg_collab_papers = s["Collaborative papers"].sum() / faculty_count
    avg_solo_papers = s["Solo papers"].sum() / faculty_count


    r = row.copy()
    r["total_papers"] = total_papers
    r["avg_papers_per_faculty"] = avg_papers
    r["log_avg_papers"] = log_avg
    r["collaboration_ratio"] = collab_ratio
    r["solo_ratio"] = solo_ratio
    r["avg_collab_papers_per_faculty"] = avg_collab_papers
    r["avg_solo_papers_per_faculty"] = avg_solo_papers
    
    rows.append(r)

out = pd.DataFrame(rows)
out.to_csv("regression_dataset.csv", index=False)

print("Saved regression_dataset.csv")
