import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_FILE = "data/ipg_metrics.csv"
OUTCOME = "pagerank"
PREDICTORS = ["EdgesPerNode", "clique_integration", "global_efficiency", "local_efficiency"]

df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=PREDICTORS + [OUTCOME])
df = df[df[OUTCOME] > 0].reset_index(drop=True)

X = sm.add_constant(df[PREDICTORS])
ols_model = sm.OLS(df[OUTCOME], X).fit()
resid = ols_model.resid
yhat = ols_model.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].hist(df[OUTCOME], bins=15, edgecolor='black', alpha=0.7)
axes[0,0].set_title(f"Distribution of {OUTCOME}\nShapiro-Wilk p={shapiro(df[OUTCOME])[1]:.4f}")
axes[0,0].set_xlabel(OUTCOME)

axes[0,1].scatter(yhat, resid, alpha=0.6)
axes[0,1].axhline(0, color='red', linestyle='--')
axes[0,1].set_title("Residuals vs Fitted")
axes[0,1].set_xlabel("Fitted values")
axes[0,1].set_ylabel("Residuals")

sm.qqplot(resid, line='s', ax=axes[1,0])
axes[1,0].set_title(f"Q-Q Plot\nShapiro-Wilk p={shapiro(resid)[1]:.4f}")

axes[1,1].hist(resid, bins=15, edgecolor='black', alpha=0.7)
axes[1,1].set_title("Distribution of Residuals")
axes[1,1].set_xlabel("Residuals")

plt.tight_layout()
plt.show()