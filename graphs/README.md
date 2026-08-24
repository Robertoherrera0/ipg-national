# Network Construction
This folder builds internal university networks from publication and grant data extracted from Dimensions.AI.

## Scripts

1. **build_internal_networks.py** - Creates adjacency matrix for each university
   - Input: Publications + Grants (XLSX), IPG faculty list (CSV), fuzzy name mappings (JSON)
   - Output: {school}_adjacency.csv in each school's graphs/ folder
   - Diagonal = solo publications (not collaborated with other IPG faculty)
   - Off-diagonal = collaborations between IPG faculty

2. **build_national_network.py** - Creates 51×51 national network matrix
   - Counts collaborations between universities' "IPG" group

3. **national_all_metrics.py** - combines internal and external metrics
