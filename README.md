# Plant Science Institutional Collaboration Networks

Pipeline for analyzing institutional-level collaboration networks from Dimensions.AI publication data across U.S. Land-Grant R1 universities (2015-2025).

## Overview

This repository provides tools to extract plant-science faculty from publication records, build institutional and national collaboration networks, compute network metrics, and perform statistical analysis on the relationships these network represent. 

## Data

Plant-science faculty are identified via fuzzy name matching, and journal affiliation. Institutional collaboration networks are built from co-authorship patterns among faculty at each university. The national network is constructed from inter-institutional co-authored publications.

## Methods

Internal network structure is measured via eight metrics including the average number of internal collaborations per faculty member, how well the largest three-clique (a group of three faculty all collaborating with each other) integrates into the school network, local and global efficiency, and assortativity measures. External research influence is measured via PageRank and other centrality measures in a pruned national network. Statistical analysis is performed to model these collaborations. 

## Contents

data/ - stores adjacency matrices for each school and the national network, plus internal and external metrics computed for every school.

graphs/ builds network structures.

pruning/ - applies thresholding to networks.

metrics/ - computes network statistics.

analysis/ - runs statistical models and exploratory analysis.

ipg_filtering/ - handles faculty identification and name standardization.
