import pandas as pd
import networkx as nx

metrics_file = "data/ipg_combined_metrics.csv"
national_file = "data/national/pruned/mutual/mutual_p0.35.csv"

df = pd.read_csv(metrics_file)

national_adj = pd.read_csv(national_file, index_col=0)
G = nx.from_pandas_adjacency(national_adj)

pagerank = nx.pagerank(G, alpha=0.85)

pagerank_df = pd.DataFrame({
    "school": list(pagerank.keys()),
    "pagerank": list(pagerank.values())
})

df = df.merge(pagerank_df, on="school")

df = df.dropna()

df.to_csv("data/ipg_metrics_with_pagerank.csv", index=False)

print("Saved: data/ipg_metrics_with_pagerank.csv")

