import os
import json
import pandas as pd
from itertools import combinations

root_directory = os.path.join(os.getcwd(), "data", "schools")

def get_school_directories():
    """Get list of all university folders."""
    schools = []
    for name in os.listdir(root_directory):
        path = os.path.join(root_directory, name)
        if not os.path.isdir(path):
            continue
        ipg_file = os.path.join(path, f"{name}_IPG.csv")
        fuzzy_file = os.path.join(path, f"{name}_fuzzy_names.json")
        if os.path.exists(ipg_file) and os.path.exists(fuzzy_file):
            schools.append(name)
    return sorted(schools)

def load_ipg_faculty(school):
    """Load IPG faculty list from CSV."""
    ipg_file = os.path.join(root_directory, school, f"{school}_IPG.csv")
    df = pd.read_csv(ipg_file)
    return set(df["Corresponding Authors"].str.strip())

def load_fuzzy_names(school):
    """Load fuzzy name mappings from JSON."""
    fuzzy_file = os.path.join(root_directory, school, f"{school}_fuzzy_names.json")
    with open(fuzzy_file, "r", encoding="utf-8") as f:
        return json.load(f)

def match_author_to_faculty(author, fuzzy_names):
    """Match an author name to canonical faculty name using fuzzy mapping."""
    author = author.strip()
    for canonical_name, variations in fuzzy_names.items():
        if author in variations:
            return canonical_name
    return None

def read_excel_safely(path):
    """Read Excel file with error handling."""
    try:
        return pd.read_excel(path)
    except Exception:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.DataFrame()

def load_publications_and_grants(school):
    """Load publication and grant data."""
    folder_path = os.path.join(root_directory, school)
    pubs_file = os.path.join(folder_path, f"{school}_Pubs.xlsx")
    grants_file = os.path.join(folder_path, f"{school}_Grants.xlsx")
    
    dataframes = []
    
    if os.path.exists(pubs_file):
        df_pubs = read_excel_safely(pubs_file)
        if not df_pubs.empty:
            dataframes.append(df_pubs)
    
    if os.path.exists(grants_file):
        df_grants = read_excel_safely(grants_file)
        if not df_grants.empty:
            dataframes.append(df_grants)
    
    if not dataframes:
        return pd.DataFrame()
    
    combined = pd.concat(dataframes, ignore_index=True)
    return combined

def main():
    schools = get_school_directories()
    print(f"Processing {len(schools)} universities\n")
    
    for school in schools:
        print(f"Building internal network for {school}...")
        
        # Load data
        ipg_faculty = load_ipg_faculty(school)
        fuzzy_names = load_fuzzy_names(school)
        publications = load_publications_and_grants(school)
        
        if publications.empty:
            print(f"  No publication data found for {school}")
            continue
        
        # Create adjacency matrix
        faculty_list = sorted(list(ipg_faculty))
        adjacency_matrix = pd.DataFrame(0, index=faculty_list, columns=faculty_list)
        
        # Process each publication/grant
        for _, row in publications.iterrows():
            authors_raw = str(row.get("Authors", ""))
            author_list = [name.strip() for name in authors_raw.split(";")]
            
            # Find which authors are IPG faculty
            ipg_authors = []
            for author in author_list:
                canonical_name = match_author_to_faculty(author, fuzzy_names)
                if canonical_name and canonical_name in ipg_faculty:
                    ipg_authors.append(canonical_name)
            
            # If only 1 IPG author, increment diagonal
            if len(ipg_authors) == 1:
                adjacency_matrix.loc[ipg_authors[0], ipg_authors[0]] += 1
            # If 2+ IPG authors, increment pairwise collaborations
            elif len(ipg_authors) > 1:
                for author_a, author_b in combinations(sorted(set(ipg_authors)), 2):
                    adjacency_matrix.loc[author_a, author_b] += 1
                    adjacency_matrix.loc[author_b, author_a] += 1
        
        # Save adjacency matrix
        output_dir = os.path.join(root_directory, school, "graphs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{school}_adjacency.csv")
        adjacency_matrix.to_csv(output_file)
        
        print(f"  Saved to {output_file}\n")
    
    print("Done.")

if __name__ == "__main__":
    main()