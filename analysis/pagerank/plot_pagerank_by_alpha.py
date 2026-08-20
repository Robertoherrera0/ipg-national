import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# last schools in ranking
# Alaska, Idaho, Maine, NewHampshire, RhodeIsland, UtahState, Vermont, WestVirginia

ranks = pd.read_csv("pagerank_ranks_by_alpha.csv", index_col=0)

x = ranks.columns.astype(float)

plt.figure(figsize=(14,10))

cmap = plt.cm.get_cmap("tab20", len(ranks))

# identify bottom row nodes (worst ranks across alphas)
bottom_nodes = ranks.max(axis=1).nlargest(8).index  # adjust 8 if needed

# create vertical spread for those nodes
offsets = np.linspace(-0.4, 0.4, len(bottom_nodes))
spread_map = dict(zip(bottom_nodes, offsets))

for i, node in enumerate(ranks.index):
    y = ranks.loc[node]
    color = cmap(i)

    plt.plot(x, y, color=color, linewidth=1, zorder=1)
    plt.scatter(x, y, color=color, s=14, zorder=2)

for xi_idx, xi in enumerate(x):

    # collect all nodes at this alpha
    y_vals = ranks.iloc[:, xi_idx]

    # group nodes by rounded rank (to detect overlaps)
    groups = {}
    for node, yi in y_vals.items():
        key = round(yi, 1)
        groups.setdefault(key, []).append(node)

    for key, nodes in groups.items():
        base_y = key

        if len(nodes) == 1:
            node = nodes[0]
            plt.text(xi, base_y, node, fontsize=6.5, color="black",
                     ha="center", va="center", zorder=3)
        else:
            offsets = np.linspace(-0.4, 0.4, len(nodes))
            for node, offset in zip(nodes, offsets):
                plt.text(
                    xi,
                    base_y + offset,
                    node,
                    fontsize=6.5,
                    color="black",
                    ha="center",
                    va="center",
                    zorder=3
                )


plt.gca().invert_yaxis()
plt.xlabel("Damping Factor (alpha)")
plt.ylabel("Rank")
plt.title("PageRank Rankings Across Damping Factors")

plt.tight_layout()
plt.savefig("pagerank_damping_factors.png", dpi=300)
plt.show()
