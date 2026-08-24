# IPG Faculty Identification
This folder identifies the plant-science faculty at each university ("IPG" - Interdisciplinary Plant Group) from raw Dimensions.AI publication records, and prepares the fuzzy name-matching data needed to build collaboration networks.

## Notebooks
1. **raw_data_treatment.ipynb**
   - Loads all papers with at least one author affiliated to the institution since 2015
   - Filters to papers where the raw affiliation matches the university/city/state
   - Removes duplicates

2. **get_plant_group.ipynb** - Identifies plant-specific faculty
   - Determines if a publication is plant-specific based in journals and keywords
   - Extracts corresponding authors from plant-specific publications
   - Fuzzy-matches author names to consolidate name variants
   - Keeps faculty with >=2 plant-specific publications as "IPG" members

3. **usda_people.ipynb** - Adds USDA-affiliated collaborators
   - Repeats the plant-specific filtering and fuzzy-matching pipeline on USDA publications associated with the institution

4. **utils.py** - Shared helper functionss