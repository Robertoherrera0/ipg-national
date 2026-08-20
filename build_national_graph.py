import pandas as pd
import networkx as nx
from pyvis.network import Network
import numpy as np
import webbrowser
import os

CSV_FILE = "national_ipg_matrix.csv"
OUTPUT_HTML = "national_ipg_network.html"

df = pd.read_csv(CSV_FILE, index_col=0)

schools = df.index.tolist()

G = nx.Graph()

# Add nodes using diagonal as total publications
for school in schools:
    total = int(df.loc[school, school])
    G.add_node(school, total=total)

# Add edges (only if weight > 0 and not diagonal)
for i, s1 in enumerate(schools):
    for j in range(i + 1, len(schools)):
        s2 = schools[j]
        weight = int(df.loc[s1, s2])
        if weight > 0:
            G.add_edge(s1, s2, weight=weight)

# Scale node sizes
totals = np.array([G.nodes[n]['total'] for n in G.nodes()])
min_t = totals.min()
max_t = totals.max()

def scale(x):
    if max_t == min_t:
        return 25
    return 15 + (x - min_t) * 45 / (max_t - min_t)

net = Network(height="900px", width="100%", bgcolor="white", font_color="black")
net.barnes_hut()

for n in G.nodes():
    size = int(scale(G.nodes[n]['total']))
    net.add_node(
        n,
        label=n,
        size=size,
        title=f"{n}<br>Total IPG Publications: {G.nodes[n]['total']}"
    )

for u, v, data in G.edges(data=True):
    net.add_edge(
        u,
        v,
        value=int(data['weight']),
        title=f"Shared Publications: {data['weight']}"
    )

net.write_html(OUTPUT_HTML)

webbrowser.open("file://" + os.path.realpath(OUTPUT_HTML))

print(f"\nOpened {OUTPUT_HTML}")
