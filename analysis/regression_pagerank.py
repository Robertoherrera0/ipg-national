import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("data/ipg_metrics_with_pagerank.csv")

y = df["pagerank"]
labels = df["school"]

predictors = [
    "EdgesPerNode",
    "clique_integration",
    "global_efficiency",
    "local_efficiency"
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, var in enumerate(predictors):
    ax = axes[i]
    x = df[var]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)

    ax.scatter(x, y)

    idx = x.argsort()
    ax.plot(x.iloc[idx], y_pred.iloc[idx], color="red")

    for j, txt in enumerate(labels):
        ax.annotate(txt, (x.iloc[j], y.iloc[j]), fontsize=6)

    ax.text(
        0.05, 0.95,
        f"β={model.params[1]:.3f}\np={model.pvalues[1]:.3f}\nR²={model.rsquared:.3f}",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )

    ax.set_title(f"{var} vs PageRank")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, var in enumerate(predictors):
    ax = axes[i]
    x = df[var]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    ax.scatter(x, residuals)

    for j, txt in enumerate(labels):
        ax.annotate(txt, (x.iloc[j], residuals.iloc[j]), fontsize=6)

    ax.axhline(0, linestyle="--")

    ax.text(
        0.05, 0.95,
        f"R²={model.rsquared:.3f}",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )

    ax.set_title(f"Residuals vs {var}")

plt.tight_layout()
plt.show()


X_full = sm.add_constant(df[predictors])
model_full = sm.OLS(y, X_full).fit()
yhat_full = model_full.fittedvalues
resid_full = model_full.resid

predictors_wo = [v for v in predictors if v != "EdgesPerNode"]
X_wo = sm.add_constant(df[predictors_wo])
model_wo = sm.OLS(y, X_wo).fit()
yhat_wo = model_wo.fittedvalues
resid_wo = model_wo.resid


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

idx = y.argsort()

axes[0].scatter(y, yhat_full)
axes[0].plot(y.iloc[idx], y.iloc[idx], color="red")

for j, txt in enumerate(labels):
    axes[0].annotate(txt, (y.iloc[j], yhat_full.iloc[j]), fontsize=6)

axes[0].text(
    0.05, 0.95,
    f"R²={model_full.rsquared:.3f}",
    transform=axes[0].transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)

axes[0].set_title("Full Model")
axes[0].set_xlabel("Actual PageRank")
axes[0].set_ylabel("Predicted PageRank")


axes[1].scatter(y, yhat_wo)
axes[1].plot(y.iloc[idx], y.iloc[idx], color="red")

for j, txt in enumerate(labels):
    axes[1].annotate(txt, (y.iloc[j], yhat_wo.iloc[j]), fontsize=6)

axes[1].text(
    0.05, 0.95,
    f"R²={model_wo.rsquared:.3f}",
    transform=axes[1].transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)

axes[1].set_title("Without EdgesPerNode")
axes[1].set_xlabel("Actual PageRank")
axes[1].set_ylabel("Predicted PageRank")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(yhat_full, resid_full)

for j, txt in enumerate(labels):
    axes[0].annotate(txt, (yhat_full.iloc[j], resid_full.iloc[j]), fontsize=6)

axes[0].axhline(0, linestyle="--")

axes[0].text(
    0.05, 0.95,
    f"R²={model_full.rsquared:.3f}",
    transform=axes[0].transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)

axes[0].set_title("Full Model Residuals")
axes[0].set_xlabel("Predicted PageRank")
axes[0].set_ylabel("Residuals")


axes[1].scatter(yhat_wo, resid_wo)

for j, txt in enumerate(labels):
    axes[1].annotate(txt, (yhat_wo.iloc[j], resid_wo.iloc[j]), fontsize=6)

axes[1].axhline(0, linestyle="--")

axes[1].text(
    0.05, 0.95,
    f"R²={model_wo.rsquared:.3f}",
    transform=axes[1].transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7)
)

axes[1].set_title("Without EdgesPerNode Residuals")
axes[1].set_xlabel("Predicted PageRank")
axes[1].set_ylabel("Residuals")

plt.tight_layout()
plt.show()