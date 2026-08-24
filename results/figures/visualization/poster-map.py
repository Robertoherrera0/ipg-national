"""
visualize_network.py
─────────────────────────────────────────────────────────────────────
US map — fixed Hawaii, edge width = weight
  - Node size  = PageRank
  - Node color = bimodal split (dark / light blue)
  - Edge width = collaboration weight (log-scaled)
─────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

ADJACENCY = "data/national/pruned/mutual/mutual_p0.35.csv"
COORDS    = "data/national/school_coordinates.csv"
METRICS   = "data/ipg_combined_metrics_v2.csv"
OUT       = "visualization/pruned/mutual"
os.makedirs(OUT, exist_ok=True)


SCHOOL_NAMES = {
    "Alaska":          "U. of Alaska",
    "Arizona":         "U. of Arizona",
    "Auburn":          "Auburn U.",
    "Clemson":         "Clemson U.",
    "ColoradoState":   "Colorado State U.",
    "Connecticut":     "U. of Connecticut",
    "Cornell":         "Cornell U.",
    "Delaware":        "U. of Delaware",
    "Florida":         "U. of Florida",
    "Georgia":         "U. of Georgia",
    "Hawaii":          "U. of Hawaii",
    "Idaho":           "U. of Idaho",
    "Illinois":        "U. of Illinois",
    "Iowa":            "U. of Iowa",
    "IowaState":       "Iowa State U.",
    "Kentucky":        "U. of Kentucky",
    "KState":          "Kansas State U.",
    "LouisianaState":  "Louisiana State U.",
    "Maine":           "U. of Maine",
    "Maryland":        "U. of Maryland",
    "Massachusetts":   "UMass Amherst",
    "Michigan":        "U. of Michigan",
    "Minnesota":       "U. of Minnesota",
    "Mississippi":     "Mississippi State U.",
    "MontanaState":    "Montana State U.",
    "MU":              "U. of Missouri",
    "Nebraska":        "U. of Nebraska",
    "Nevada":          "U. of Nevada",
    "NewHampshire":    "U. of New Hampshire",
    "NewMexico":       "New Mexico State U.",
    "NorthCarolina":   "NC State U.",
    "NorthDakota":     "North Dakota State U.",
    "OhioState":       "Ohio State U.",
    "OKState":         "Oklahoma State U.",
    "Oregon":          "Oregon State U.",
    "PennState":       "Penn State U.",
    "Purdue":          "Purdue U.",
    "RhodeIsland":     "U. of Rhode Island",
    "Rutgers":         "Rutgers U.",
    "SouthDakota":     "South Dakota State U.",
    "TexasA&M":        "Texas A&M U.",
    "UArk":            "U. of Arkansas",
    "UCDavis":         "UC Davis",
    "UCRiver":         "UC Riverside",
    "UtahState":       "Utah State U.",
    "UTenn":           "U. of Tennessee",
    "Vermont":         "U. of Vermont",
    "VirginiaTech":    "Virginia Tech",
    "WashState":       "Washington State U.",
    "WestVirginia":    "West Virginia U.",
    "Wisconsin":       "U. of Wisconsin",
    "Wyoming":         "U. of Wyoming",
}
# ── D
def display_name(school):
    return SCHOOL_NAMES.get(school, school)

# ── LOAD ──────────────────────────────────────────────────────────
adj     = pd.read_csv(ADJACENCY, index_col=0)
adj.index   = adj.index.str.strip()
adj.columns = adj.columns.str.strip()

coords  = pd.read_csv(COORDS).drop_duplicates(subset="school").set_index("school")
coords.index = coords.index.str.strip()

coords.loc["Hawaii", "latitude"]  = 26.0
coords.loc["Hawaii", "longitude"] = -106.0

metrics = pd.read_csv(METRICS).dropna(subset=["pagerank", "EdgesPerNode"])
metrics["school"] = metrics["school"].str.strip()

pr_thresh    = metrics["pagerank"].min() * 1.05
connected    = metrics[metrics["pagerank"] > pr_thresh].copy()
disconnected = metrics[metrics["pagerank"] <= pr_thresh]["school"].tolist()
pr_vals      = connected["pagerank"].values

# bimodal split
kde      = gaussian_kde(pr_vals, bw_method=0.3)
x_range  = np.linspace(pr_vals.min(), pr_vals.max(), 1000)
density  = kde(x_range)
mid_mask = (x_range > np.percentile(pr_vals, 20)) & \
           (x_range < np.percentile(pr_vals, 80))
valley_x = x_range[mid_mask][np.argmin(density[mid_mask])]
connected["group"] = np.where(connected["pagerank"] >= valley_x,
                               "Upper mode", "Lower mode")

pr_map    = dict(zip(connected["school"], connected["pagerank"]))
epn_map   = dict(zip(connected["school"], connected["EdgesPerNode"]))
group_map = dict(zip(connected["school"], connected["group"]))

print(f"  Split at {valley_x:.5f}")
print(f"  Upper: {sorted(connected[connected['group']=='Upper mode']['school'].tolist())}")
print(f"  Lower: {sorted(connected[connected['group']=='Lower mode']['school'].tolist())}")

# ── GRAPH ─────────────────────────────────────────────────────────
G = nx.from_pandas_adjacency(adj)
G.remove_edges_from(nx.selfloop_edges(G))
schools_with_coords = set(coords.index)

pr_min, pr_max = min(pr_map.values()), max(pr_map.values())
SIZE_MIN, SIZE_MAX = 8, 45

def scale_size(v):
    return SIZE_MIN + (v - pr_min) / (pr_max - pr_min) * (SIZE_MAX - SIZE_MIN)

# ── Validate coordinates ──────────────────────────────────────────
for school in list(schools_with_coords):
    lat = coords.loc[school, "latitude"]
    lon = coords.loc[school, "longitude"]
    if pd.isna(lat) or pd.isna(lon):
        print(f"  WARNING: dropping {school} — missing coords")
        schools_with_coords.discard(school)
    elif not (-180 <= lon <= -60 and 15 <= lat <= 72):
        print(f"  WARNING: dropping {school} — coords out of range ({lat}, {lon})")
        schools_with_coords.discard(school)

# ── FIGURE ────────────────────────────────────────────────────────
fig = go.Figure()

# ── EDGES (log-scaled width) ──────────────────────────────────────
edge_weights = [G[u][v].get("weight", 1)
                for u, v in G.edges()
                if u in schools_with_coords and v in schools_with_coords]
w_min = min(edge_weights) if edge_weights else 1
w_max = max(edge_weights) if edge_weights else 1

WIDTH_MIN, WIDTH_MAX = 0.5, 6.0

print(f"  Weight range: {w_min} – {w_max}")

for u, v in G.edges():
    if u not in schools_with_coords or v not in schools_with_coords:
        continue

    lat1 = float(coords.loc[u, "latitude"])
    lon1 = float(coords.loc[u, "longitude"])
    lat2 = float(coords.loc[v, "latitude"])
    lon2 = float(coords.loc[v, "longitude"])

    w    = G[u][v].get("weight", 1)
    frac = ((w - w_min) / (w_max - w_min)) ** 2 if w_max > w_min else 0.5
    width   = WIDTH_MIN + frac * (WIDTH_MAX - WIDTH_MIN)
    opacity = 0.45 + 0.55 * frac

    fig.add_trace(go.Scattergeo(
        lon=[lon1, lon2],
        lat=[lat1, lat2],
        mode="lines",
        line=dict(width=width, color=f"rgba(30,50,110,{opacity:.2f})"),
        hoverinfo="skip",
        showlegend=False,
    ))

# ── NODES ─────────────────────────────────────────────────────────
for group, color, name in [
    ("Upper mode", "#1A3A5C", "High PageRank"),
    ("Lower mode", "#7FAFD4", "Low PageRank"),
]:
    lats, lons, sizes, texts, hovers = [], [], [], [], []
    for school, grp in group_map.items():
        if grp != group or school not in schools_with_coords:
            continue
        pr_val = pr_map[school]
        epn    = epn_map.get(school, np.nan)
        lats.append(float(coords.loc[school, "latitude"]))
        lons.append(float(coords.loc[school, "longitude"]))
        sizes.append(scale_size(pr_val))
        texts.append(display_name(school))          # ← mapped name
        hovers.append(
            f"<b>{display_name(school)}</b><br>"
            f"PageRank: {pr_val:.5f}<br>"
            f"EdgesPerNode: {epn:.3f}"
        )
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats,
        mode="markers+text",
        text=texts,
        textposition="top center",
        textfont=dict(size=8, color="#1a1a1a"),
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(size=sizes, color=color, opacity=0.88,
                    line=dict(width=1.2, color="white")),
        name=name,
        showlegend=True,
    ))

# ── DISCONNECTED ──────────────────────────────────────────────────
disc_lats, disc_lons, disc_texts, disc_hovers = [], [], [], []
for school in disconnected:
    if school not in schools_with_coords:
        continue
    disc_lats.append(float(coords.loc[school, "latitude"]))
    disc_lons.append(float(coords.loc[school, "longitude"]))
    disc_texts.append(display_name(school))         # ← mapped name
    disc_hovers.append(f"<b>{display_name(school)}</b><br>Not connected")

if disc_lats:
    fig.add_trace(go.Scattergeo(
        lon=disc_lons, lat=disc_lats,
        mode="markers+text",
        text=disc_texts,
        textposition="top center",
        textfont=dict(size=7, color="#999"),
        hovertext=disc_hovers,
        hoverinfo="text",
        marker=dict(size=5, color="#D5D8DC", opacity=0.55,
                    line=dict(width=0.8, color="#AAA")),
        name="Not connected",
        showlegend=True,
    ))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    geo=dict(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#F4F6F7",
        showlakes=True,
        lakecolor="#D6EAF8",
        showcoastlines=True,
        coastlinecolor="#BBBBBB",
        showframe=False,
        showsubunits=True,
        subunitcolor="#DDDDDD",
        bgcolor="rgba(0,0,0,0)",
    ),
    legend=dict(
        title=dict(text="PageRank tier", font=dict(size=12)),
        x=0.01, y=0.97,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#CCCCCC",
        borderwidth=1,
        font=dict(size=11),
    ),
    width=1500,
    height=900,
    margin=dict(l=0, r=0, t=80, b=0),
)

out_path = os.path.join(OUT, "network_pagerank_map.html")
fig.write_html(out_path)
print(f"\n  {out_path}")

out_path = os.path.join(OUT, "network_pagerank_map.svg")
fig.write_image(out_path)
print(f"  {out_path}")