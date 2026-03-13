import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv("data/ipg_combined_metrics.csv")

predictors = [
    "EdgesPerNode",
    "assortativity_unweighted",
    "global_efficiency",
    "local_efficiency",
    "avg_path_over_diameter",
    "clique_integration",
    "sigma_small_world_index",
    "omega_small_world_index"
]

y = df["pagerank"]
X = df[predictors]

# ---------------------------------------------------------
# OLS regression
# ---------------------------------------------------------

X_const = sm.add_constant(X)

model = sm.OLS(y, X_const).fit()

print("\nOLS REGRESSION: PageRank ~ Internal Metrics\n")
print(model.summary())

# ---------------------------------------------------------
# Standardized coefficients
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
y_scaled = (y - y.mean()) / y.std()

model_std = sm.OLS(y_scaled, sm.add_constant(X_scaled)).fit()

std_coefs = pd.Series(model_std.params[1:], index=predictors)

print("\nSTANDARDIZED COEFFICIENTS\n")
print(std_coefs.sort_values(key=abs, ascending=False))

# ---------------------------------------------------------
# Variance Inflation Factor (multicollinearity)
# ---------------------------------------------------------

vif_df = pd.DataFrame()
vif_df["variable"] = predictors
vif_df["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\nVARIANCE INFLATION FACTOR\n")
print(vif_df.sort_values("VIF", ascending=False))