# Statistical Analysis
This folder contains all statistical modeling and analysis linking internal network structure to external (national) network position.

## Subfolders

### regression
Core statistical models.
- **gamma_glm.py** - Shared Gamma GLM (log link) fitting
- **pagerank_regression.py** - Internal metrics to PageRank. Fits OLS and Gamma GLM.
- **intra_inter_regression.py** - Internal metrics -> all other external centrality metrics (degree, eigenvector, closeness, betweenness, clustering).

### pagerank
PageRank-specific robustness checks.
- **damping.py** - Tests whether school rankings are stable across a range of PageRank damping factors
- **pagerank_analysis.py** - Computes PageRank and correlates it with other centrality metrics

### pca
Exploratory dimensionality reduction on internal network metrics.
- **national_pca.py** - PCA on internal metrics across all schools

## Input Data
All scripts read from:
- `data/ipg_metrics.csv` - merged internal + external metrics
- `data/national/school_internal_metrics.csv` - full internal metrics
- `data/national/pruned/mutual/mutual_p0.35.csv` - pruned national network
  national network
