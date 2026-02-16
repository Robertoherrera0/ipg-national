import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("regression_dataset.csv").replace([np.inf, -np.inf], np.nan).dropna()

predictors = [
    "collaboration_ratio",
    "EdgesPerNode",
    "global_efficiency",
    "assortativity_weighted",
    "clique_integration",
    "local_efficiency",
    "sigma_small_world_index",
    "omega_small_world_index"
]

X = df[predictors].values
y = df["log_avg_papers"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ols = LinearRegression().fit(X_scaled, y)
ridge = Ridge(alpha=1.0).fit(X_scaled, y)

df_plot = df.copy()
df_plot["Actual"] = y
df_plot["Pred_OLS"] = ols.predict(X_scaled)
df_plot["Pred_Ridge"] = ridge.predict(X_scaled)

xmin, xmax = df_plot["Actual"].min(), df_plot["Actual"].max()
pad = 0.05 * (xmax - xmin)

fig_ols = px.scatter(
    df_plot,
    x="Actual",
    y="Pred_OLS",
    color="Group",
    text="School",
    hover_name="School",
    hover_data=predictors
)

fig_ols.update_traces(
    mode="markers+text",
    textposition="top right",
    textfont=dict(size=8),
    cliponaxis=False
)

fig_ols.update_layout(
    title=f"OLS Actual vs Predicted (R² = {ols.score(X_scaled, y):.3f})",
    xaxis_title="Actual log(avg papers)",
    yaxis_title="Predicted log(avg papers)",
    xaxis=dict(range=[xmin - pad, xmax + pad]),
    yaxis=dict(range=[xmin - pad, xmax + pad])
)

fig_ols.add_shape(
    type="line",
    x0=xmin,
    y0=xmin,
    x1=xmax,
    y1=xmax,
    line=dict(color="red", dash="dash")
)

fig_ols.show()

fig_ridge = px.scatter(
    df_plot,
    x="Actual",
    y="Pred_Ridge",
    color="Group",
    text="School",
    hover_name="School",
    hover_data=predictors
)

fig_ridge.update_traces(
    mode="markers+text",
    textposition="top right",
    textfont=dict(size=8),
    cliponaxis=False
)

fig_ridge.update_layout(
    title=f"Ridge (alpha=1.0) Actual vs Predicted (R² = {ridge.score(X_scaled, y):.3f})",
    xaxis_title="Actual log(avg papers)",
    yaxis_title="Predicted log(avg papers)",
    xaxis=dict(range=[xmin - pad, xmax + pad]),
    yaxis=dict(range=[xmin - pad, xmax + pad])
)

fig_ridge.add_shape(
    type="line",
    x0=xmin,
    y0=xmin,
    x1=xmax,
    y1=xmax,
    line=dict(color="red", dash="dash")
)

fig_ridge.show()

coef_df = pd.DataFrame({
    "Variable": predictors,
    "OLS": ols.coef_,
    "Ridge": ridge.coef_
})

fig_coef = go.Figure()
fig_coef.add_bar(x=coef_df["Variable"], y=coef_df["OLS"], name="OLS")
fig_coef.add_bar(x=coef_df["Variable"], y=coef_df["Ridge"], name="Ridge")
fig_coef.update_layout(
    barmode="group",
    title="Standardized Coefficients",
    yaxis_title="Coefficient"
)
fig_coef.show()

# imp_df = pd.DataFrame({
#     "Variable": predictors,
#     "OLS": np.abs(ols.coef_),
#     "Ridge": np.abs(ridge.coef_)
# }).sort_values("Ridge")

# fig_imp = go.Figure()
# fig_imp.add_bar(y=imp_df["Variable"], x=imp_df["OLS"], orientation="h", name="OLS")
# fig_imp.add_bar(y=imp_df["Variable"], x=imp_df["Ridge"], orientation="h", name="Ridge")
# fig_imp.update_layout(
#     barmode="group",
#     title="Relative Importance",
#     xaxis_title="|Standardized coefficient|"
# )
# fig_imp.show()