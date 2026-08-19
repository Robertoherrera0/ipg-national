import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/ipg_combined_metrics_v2.csv")

# Remove low PageRank outliers (your filter)
pr_thresh = df["pagerank"].min() * 1.05
df = df[df["pagerank"] > pr_thresh].copy().reset_index(drop=True)

# Define metrics
internal_metrics = [
    "EdgesPerNode", "clique_integration", "global_efficiency", 
    "local_efficiency", "pagerank", "degree_centrality", 
    "eigenvector_centrality", "closeness_centrality"
]

external_metrics = [
    "pagerank", "degree_centrality", "eigenvector_centrality",
    "closeness_centrality", "betweenness_centrality", "clustering"
]

# Actually wait - YOUR COLUMNS HAVE OVERLAP
# Let me check your actual internal vs external split