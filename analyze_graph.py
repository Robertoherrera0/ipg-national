import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.smallworld import sigma as _sigma, omega as _omega
from networkx.algorithms.community import k_clique_communities
from openpyxl import load_workbook

def load_adjacency_matrix(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.astype(str)
    df.index = df.index.astype(str)
    G = nx.Graph()
    for node in df.index:
        G.add_node(node)
    for i, r in enumerate(df.index):
        for j, c in enumerate(df.columns):
            if j <= i:
                continue
            if df.iloc[i, j] > 0:
                G.add_edge(r, c, weight=float(df.iloc[i, j]))
    return G

def safe(f, default=""):
    try:
        return f()
    except:
        return default

def compute_clique_metrics(G, k=3):
    comms = list(k_clique_communities(G, k))
    n_nodes = G.number_of_nodes()
    n_comms = len(comms)
    if n_comms == 0:
        return 0, 0, 0
    largest_size = max(len(c) for c in comms)
    integration = largest_size / n_nodes if n_nodes > 0 else 0
    return n_comms, largest_size, integration

def compute_path_metrics(G):
    try:
        if nx.is_connected(G):
            H = G
        else:
            gc = max(nx.connected_components(G), key=len)
            H = G.subgraph(gc)
        avg = nx.average_shortest_path_length(H)
        diam = nx.diameter(H)
        ratio = avg / diam if diam > 0 else 0
        return avg, diam, ratio
    except:
        return "", "", ""

def compute_efficiency(G):
    return safe(lambda: nx.global_efficiency(G)), safe(lambda: nx.local_efficiency(G))

def compute_assortativity(G):
    return safe(lambda: nx.degree_assortativity_coefficient(G)), \
           safe(lambda: nx.degree_assortativity_coefficient(G, weight="weight"))

def compute_smallworld(G):
    try:
        if nx.is_connected(G):
            H = G.copy()
        else:
            gc = max(nx.connected_components(G), key=len)
            H = G.subgraph(gc).copy()
        sig = _sigma(H, niter=20, nrand=10)
        omg = _omega(H, niter=20, nrand=10)
        return sig, omg
    except:
        return "", ""

def excel_column(path):
    wb = load_workbook(path)
    ws = wb.active
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                val = str(cell.value)
            except:
                val = ""
            if val is None:
                val = ""
            if len(val) > max_len:
                max_len = len(val)
        ws.column_dimensions[col_letter].width = max_len + 2
    wb.save(path)

def main():
    root = os.getcwd()
    schools = [
        s for s in os.listdir(root)
        if os.path.isdir(os.path.join(root, s)) and s not in {"lib", "__pycache__"}
    ]

    rows = []

    for school in schools:
        adj = os.path.join(root, school, "graphs", f"{school}_adjacency.csv")
        if not os.path.exists(adj):
            continue

        G = safe(lambda: load_adjacency_matrix(adj), None)
        if G is None:
            continue

        n = G.number_of_nodes()
        e = G.number_of_edges()
        epn = e / n if n > 0 else 0

        n3, largest, integ = compute_clique_metrics(G, k=3)
        avg, diam, ratio = compute_path_metrics(G)
        ge, le = compute_efficiency(G)
        au, aw = compute_assortativity(G)
        s, o = compute_smallworld(G)

        rows.append({
            "School": school,
            "Nodes": n,
            "Edges": e,
            "EdgesPerNode": epn,
            "assortativity_unweighted": au,
            "assortativity_weighted": aw,
            "average_shortest_path_length": avg,
            "diameter": diam,
            "avg_path_over_diameter": ratio,
            "n_cliques": n3,
            "largest_clique_size": largest,
            "clique_integration": integ,
            "global_efficiency": ge,
            "local_efficiency": le,
            "sigma_small_world_index": s,
            "omega_small_world_index": o
        })

    out = "IPG_metrics2.xlsx"
    df = pd.DataFrame(rows)
    df.to_excel(out, index=False)
    excel_column(out)
    print("Saved IPG_metrics2.xlsx")

if __name__ == "__main__":
    main()
