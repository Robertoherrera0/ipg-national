# Network Pruning
This folder prunes the dense national collaboration network into a 
sparser, interpretable structure by keeping only the strongest ties 
between schools.

## Scripts

1. **mutual-pruning.py** 
   - Keeps an edge between two schools only if each is in the other's top N% of collaborators (by publication count)
   - Sweeps thresholds: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35

2. **threshold-pruning.py**
   - Keeps an edge only if the relative collaboration share exceeds a percentage threshold AND the raw publication count exceeds a floor
   - Sweeps combinations of thresholds (0.03–0.25) and floors (20–60)

## Why Pruning Is Necessary

The raw national network is a hairball. Nearly every school has at least one publication in common with nearly every other school. Pruning removes weak ties to reveal the structurally meaningful  collaborations.