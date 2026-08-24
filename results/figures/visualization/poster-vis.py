import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

df = pd.read_csv("data/ipg_combined_metrics_v2.csv")
df = df.dropna(subset=["pagerank", "EdgesPerNode"]).copy()
df["school"] = df["school"].str.strip()

pr_thresh = df["pagerank"].min() * 1.05
connected = df[df["pagerank"] > pr_thresh].copy()

pr_vals = connected["pagerank"].values
kde = gaussian_kde(pr_vals, bw_method=0.3)
x_range = np.linspace(pr_vals.min(), pr_vals.max(), 1000)
density = kde(x_range)
mid_mask = (
    (x_range > np.percentile(pr_vals, 20)) &
    (x_range < np.percentile(pr_vals, 80))
)
valley_x = x_range[mid_mask][np.argmin(density[mid_mask])]
connected["group"] = np.where(
    connected["pagerank"] >= valley_x,
    "Higher national influence",
    "Lower national influence"
)

SCHOOL_NAMES = {
    "Alaska": "U. of Alaska", "Arizona": "U. of Arizona", "Auburn": "Auburn U.",
    "Clemson": "Clemson U.", "ColoradoState": "Colorado State U.",
    "Connecticut": "U. of Connecticut", "Cornell": "Cornell U.",
    "Delaware": "U. of Delaware", "Florida": "U. of Florida",
    "Georgia": "U. of Georgia", "Hawaii": "U. of Hawaii", "Idaho": "U. of Idaho",
    "Illinois": "U. of Illinois", "Iowa": "U. of Iowa", "IowaState": "Iowa State U.",
    "Kentucky": "U. of Kentucky", "KState": "Kansas State U.",
    "LouisianaState": "Louisiana State U.", "Maine": "U. of Maine",
    "Maryland": "U. of Maryland", "Massachusetts": "UMass Amherst",
    "Michigan": "U. of Michigan", "Minnesota": "U. of Minnesota",
    "Mississippi": "Mississippi State U.", "MontanaState": "Montana State U.",
    "MU": "U. of Missouri", "Nebraska": "U. of Nebraska", "Nevada": "U. of Nevada",
    "NewHampshire": "U. of New Hampshire", "NewMexico": "New Mexico State U.",
    "NorthCarolina": "NC State U.", "NorthDakota": "North Dakota State U.",
    "OhioState": "Ohio State U.", "OKState": "Oklahoma State U.",
    "Oregon": "Oregon State U.", "PennState": "Penn State U.", "Purdue": "Purdue U.",
    "RhodeIsland": "U. of Rhode Island", "Rutgers": "Rutgers U.",
    "SouthDakota": "South Dakota State U.", "TexasA&M": "Texas A&M U.",
    "UArk": "U. of Arkansas", "UCDavis": "UC Davis", "UCRiver": "UC Riverside",
    "UTenn": "U. of Tennessee", "UtahState": "Utah State U.",
    "Vermont": "U. of Vermont", "VirginiaTech": "Virginia Tech",
    "WashState": "Washington State U.", "WestVirginia": "West Virginia U.",
    "Wisconsin": "U. of Wisconsin", "Wyoming": "U. of Wyoming",
}

connected["display"] = connected["school"].map(lambda x: SCHOOL_NAMES.get(x, x))
connected["avg_collab"] = connected["EdgesPerNode"]
connected = connected.sort_values("pagerank", ascending=False).reset_index(drop=True)

high_mask = connected["group"] == "Higher national influence"
low_mask  = connected["group"] == "Lower national influence"
high_min = connected.loc[high_mask, "pagerank"].min()
high_max = connected.loc[high_mask, "pagerank"].max()
low_min  = connected.loc[low_mask, "pagerank"].min()
low_max  = connected.loc[low_mask, "pagerank"].max()

def blend_rgb(c1, c2, t):
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"rgb({r},{g},{b})"

LOW_START = (190, 220, 240)
LOW_END   = (127, 175, 212)
HIGH_START = (90, 130, 170)
HIGH_END   = (26, 58, 92)

def assign_color(row):
    pr = row["pagerank"]
    grp = row["group"]
    if grp == "Higher national influence":
        t = 0.0 if high_max == high_min else (pr - high_min) / (high_max - high_min)
        return blend_rgb(HIGH_START, HIGH_END, t)
    else:
        t = 0.0 if low_max == low_min else (pr - low_min) / (low_max - low_min)
        return blend_rgb(LOW_START, LOW_END, t)

connected["bar_color"] = connected.apply(assign_color, axis=1)

LEGEND_HIGH = "rgb(40,80,120)"
LEGEND_LOW  = "rgb(150,195,225)"

fig = go.Figure()

fig.add_trace(go.Bar(
    x=connected["display"].tolist(),
    y=connected["avg_collab"].tolist(),
    marker=dict(color=connected["bar_color"].tolist(), line=dict(width=0)),
    customdata=np.stack([connected["pagerank"].to_numpy(), connected["group"].to_numpy()], axis=-1),
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Average number of collaborators per faculty member: %{y:.3f}<br>"
        "PageRank: %{customdata[0]:.5f}<br>"
        "Influence group: %{customdata[1]}<extra></extra>"
    ),
    showlegend=False
))

fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=LEGEND_HIGH),
                     name="Higher national influence", showlegend=True))
fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=LEGEND_LOW),
                     name="Lower national influence", showlegend=True))

fig.update_xaxes(
    title_text="R1 Land-Grant Universities",
    tickangle=-30,
    categoryorder="array",
    categoryarray=connected["display"].tolist(),
    tickfont=dict(size=11),
    title_font=dict(size=18)
)

fig.update_yaxes(
    title_text="Average Number of Collaborators per Faculty Member",
    showgrid=True,
    gridcolor="rgba(200,200,200,0.35)",
    zeroline=False,
    tickfont=dict(size=14),
    title_font=dict(size=18)
)

fig.update_layout(
    title=dict(text=""),
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.02,
        yanchor="bottom",
        font=dict(size=22)
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    width=1700,
    height=650,
    margin=dict(l=120, r=120, t=80, b=180),
    font=dict(family="Arial", color="black"),
    barmode="group"
)

fig.show()
fig.write_image("figures/fig_ranking_side_by_side_same_groups.svg", width=1700, height=650)