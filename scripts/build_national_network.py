import os
import json
import pandas as pd
from itertools import combinations

root_directory = os.getcwd()


def get_school_directories():
    schools = []
    for name in os.listdir(root_directory):
        path = os.path.join(root_directory, name)
        if not os.path.isdir(path):
            continue

        publications_file = os.path.join(path, f"{name}_Pubs.csv")
        fuzzy_file = os.path.join(path, f"{name}_fuzzy_names.json")

        if os.path.exists(publications_file) and os.path.exists(fuzzy_file):
            schools.append(name)

    return sorted(schools)


def load_ipg_variants(school):
    file_path = os.path.join(root_directory, school, f"{school}_fuzzy_names.json")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    variants = set()
    for values in data.values():
        for name in values:
            variants.add(name.strip())

    return variants


def read_csv_safely(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def load_publications(school):
    folder_path = os.path.join(root_directory, school)

    main_file = os.path.join(folder_path, f"{school}_Pubs.csv")
    usda_file = os.path.join(folder_path, f"{school}_plus_USDA_Pubs.csv")

    dataframes = []

    if os.path.exists(main_file):
        df_main = read_csv_safely(main_file)
        if not df_main.empty:
            dataframes.append(df_main)

    if os.path.exists(usda_file):
        df_usda = read_csv_safely(usda_file)
        if not df_usda.empty:
            dataframes.append(df_usda)

    if not dataframes:
        return pd.DataFrame()

    combined = pd.concat(dataframes, ignore_index=True)

    if "Publication ID" in combined.columns:
        combined = combined.drop_duplicates(subset="Publication ID")

    return combined


def main():

    schools = get_school_directories()
    if "Iowa" in schools:
        schools.remove("Iowa")
    print(f"Detected {len(schools)} schools\n")

    school_variants = {}
    all_publications = {}

    for school in schools:
        print(f"Processing {school}")

        variants = load_ipg_variants(school)
        school_variants[school] = variants

        publications = load_publications(school)
        print(f"Total publications loaded: {len(publications)}")

        all_publications[school] = publications

    publication_to_schools = {}

    print("\nScanning publications for IPG co-authorship...\n")

    for school in schools:

        dataframe = all_publications[school]
        variants = school_variants[school]

        if dataframe.empty:
            continue

        for _, row in dataframe.iterrows():

            publication_id = row.get("Publication ID")
            authors_raw = str(row.get("Authors", ""))

            author_list = [name.strip() for name in authors_raw.split(";")]

            found = False
            for author in author_list:
                if author in variants:
                    found = True
                    break

            if found:
                if publication_id not in publication_to_schools:
                    publication_to_schools[publication_id] = set()

                publication_to_schools[publication_id].add(school)

    matrix = pd.DataFrame(0, index=schools, columns=schools)

    for publication_id, school_set in publication_to_schools.items():

        if len(school_set) < 2:
            continue

        for school_a, school_b in combinations(sorted(school_set), 2):
            matrix.loc[school_a, school_b] += 1
            matrix.loc[school_b, school_a] += 1

    for school in schools:
        count = sum(
            1 for schools_on_pub in publication_to_schools.values()
            if school in schools_on_pub
        )
        matrix.loc[school, school] = count

    matrix.to_csv("national_ipg_matrix.csv")

    print("done.")


if __name__ == "__main__":
    main()
