import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

df = pd.read_csv("data/ipg_combined_metrics_v2.csv")
pr_thresh = df["pagerank"].min() * 1.05
df = df[df["pagerank"] > pr_thresh].copy().reset_index(drop=True)
df["log_EPN"] = np.log1p(df["EdgesPerNode"])

pr_vals = df["pagerank"].values
kde = gaussian_kde(pr_vals, bw_method=0.3)
x_range = np.linspace(pr_vals.min(), pr_vals.max(), 1000)
density = kde(x_range)
mid_mask = (
    (x_range > np.percentile(pr_vals, 20)) &
    (x_range < np.percentile(pr_vals, 80))
)
valley_x = x_range[mid_mask][np.argmin(density[mid_mask])]
df["group"] = np.where(df["pagerank"] >= valley_x, "High PageRank", "Low PageRank")

m_gamma = smf.glm("pagerank ~ log_EPN", data=df,
                   family=sm.families.Gamma(link=sm.families.links.Log())).fit()
r2 = 1 - m_gamma.deviance / m_gamma.null_deviance
beta = m_gamma.params["log_EPN"]
pval = m_gamma.pvalues["log_EPN"]

xfit = np.linspace(df["log_EPN"].min(), df["log_EPN"].max(), 200)
pred = m_gamma.get_prediction(pd.DataFrame({"log_EPN": xfit}))
pred_summary = pred.summary_frame(alpha=0.05)
yfit  = pred_summary["mean"].values
lower = pred_summary["mean_ci_lower"].values
upper = pred_summary["mean_ci_upper"].values

colors = ["#1A3A5C" if g == "High PageRank" else "#85B7EB" for g in df["group"]]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.concatenate([xfit, xfit[::-1]]),
    y=np.concatenate([upper, lower[::-1]]),
    fill="toself",
    fillcolor="rgba(192,57,43,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="none",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=xfit, y=yfit,
    mode="lines",
    line=dict(color="#C0392B", width=2.5),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=df["log_EPN"],
    y=df["pagerank"],
    mode="markers+text",
    marker=dict(color=colors, size=10,
                line=dict(color="white", width=0.5)),
    text=df["school"],
    textposition="top right",
    textfont=dict(size=7, color="rgba(50,50,50,0.75)"),
    hovertemplate="<b>%{text}</b><br>EdgesPerNode: %{x:.2f}<br>PageRank: %{y:.4f}<extra></extra>",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(color="#1A3A5C", size=12),
    name="High PageRank"
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(color="#85B7EB", size=12),
    name="Low PageRank"
))

fig.add_annotation(
    x=0.02, y=0.80,
    xref="paper", yref="paper",
    text=f"Gamma GLM — log link<br>β = {beta:.4f}<br>p < 0.0001<br>Pseudo R² = {r2:.3f}",
    showarrow=False,
    align="left",
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="#1A3A5C",
    borderwidth=1,
    font=dict(size=14, color="#1A3A5C")
)

fig.update_layout(
    title=dict(text=""),
    xaxis=dict(
        title="Average Number of Collaborators per Faculty Member",
        showgrid=False,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title="National Influence (PageRank)",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    width=1000,
    height=700,
    legend=dict(x=0.02, y=0.98, font=dict(size=14))
)

fig.write_image("figures/gamma_poster.svg")
fig.write_html("figures/gamma_poster.html")
fig.show()