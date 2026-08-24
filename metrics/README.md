# Network Metrics
This folder computes all network metrics used in the statistical analysis, 
from the internal (within each school) and national collaboration networks.

## Scripts
1. **internal_metrics.py** - Computes 8 internal network metrics per university
   - Metrics: EdgesPerNode, assortativity (weighted/unweighted), 
     avg_path_over_diameter, clique_integration, global_efficiency, 
     local_efficiency, sigma (small-world index), omega (small-world index)

2. **national_ipg_metrics.py** - Computes external network metrics from the 
   pruned national network
   - Metrics: pagerank, degree_centrality, weighted_degree, eigenvector_centrality, closeness_centrality, betweenness_centrality, clustering
   - All measures weighted by collaboration count where supported

3. **publication_stats.py** - Computes internal vs external publication ratios 
   per university 