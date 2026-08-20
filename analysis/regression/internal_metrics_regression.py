import pandas as pd
import networkx as nx

# load adjacency matrix
adj = pd.read_csv("data/national/pruned/mutual/mutual_p0.35.csv", index_col=0)

# build graph from adjacency matrix
Gnat = nx.from_pandas_adjacency(adj)

alphas = [0.5, 0.65, 0.75, 0.85, 0.90, 0.95]

results = {}

for a in alphas:
    pr = nx.pagerank(Gnat, alpha=a, weight="weight")
    results[f"alpha_{a}"] = pr

df = pd.DataFrame(results)
df.index.name = "institution"

rank_df = df.rank(ascending=False)

print("\nTop institutions (alpha=0.85)\n")
print(rank_df.sort_values("alpha_0.85").head(20))