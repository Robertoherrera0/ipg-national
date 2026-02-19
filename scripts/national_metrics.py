import os
import pandas as pd
import networkx as nx

folder_path = os.path.expanduser("~/Downloads/NationalPG")

network_files = network_files = [
    "national_ipg_matrix",
    "national_ipg_matrix_relative",
    "national_ipg_matrix_unweighted"
]

output_file = os.path.join(folder_path, "network_metrics.xlsx")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    for network_name in network_files:
        file_path = os.path.join(folder_path, network_name + ".csv")
        
        df = pd.read_csv(file_path, index_col=0)
        
        G = nx.from_pandas_adjacency(df)
        
        degree_centrality = nx.degree_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        clustering_coefficient = nx.clustering(G)
        
        voterank_nodes = nx.voterank(G)
        voterank_scores = {node: 0 for node in G.nodes()}
        for rank, node in enumerate(voterank_nodes):
            voterank_scores[node] = len(voterank_nodes) - rank
        
        results = pd.DataFrame({
            "Degree Centrality": pd.Series(degree_centrality),
            "Eigenvector Centrality": pd.Series(eigenvector_centrality),
            "Closeness Centrality": pd.Series(closeness_centrality),
            "Betweenness Centrality": pd.Series(betweenness_centrality),
            "Clustering Coefficient": pd.Series(clustering_coefficient),
            "VoteRank Score": pd.Series(voterank_scores)
        })
        
        results.to_excel(writer, sheet_name=network_name)

print(f"Yay! saved to: {output_file}")

