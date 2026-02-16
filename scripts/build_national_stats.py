import os
import pandas as pd
import numpy as np

root_directory = os.getcwd()
national_matrix_file = "national_ipg_matrix.csv"


def main():

    if not os.path.exists(national_matrix_file):
        print("national_ipg_matrix.csv not found.")
        return

    matrix = pd.read_csv(national_matrix_file, index_col=0)

    schools = matrix.index.tolist()

    results = []

    print("\nAnalyzing national IPG matrix...\n")

    for school in schools:

        total_ipg_publications = int(matrix.loc[school, school])

        row = matrix.loc[school].drop(school)

        partner_schools = row[row > 0]

        number_of_partner_schools = len(partner_schools)

        total_pairwise_external_publications = int(partner_schools.sum())

        if total_ipg_publications > 0:
            external_ratio = total_pairwise_external_publications / total_ipg_publications
            internal_ratio = 1 - external_ratio
        else:
            external_ratio = 0
            internal_ratio = 0

        results.append({
            "School": school,
            "Total_IPG_Publications": total_ipg_publications,
            "Number_of_Partner_Schools": number_of_partner_schools,
            "Total_Pairwise_External_Publications": total_pairwise_external_publications,
            "External_Publication_Ratio": round(external_ratio, 4),
            "Internal_Publication_Ratio": round(internal_ratio, 4)
        })

        print(f"{school}")
        print(f"  Total IPG Publications: {total_ipg_publications}")
        print(f"  Partner Schools: {number_of_partner_schools}")
        print(f"  Pairwise External Publications: {total_pairwise_external_publications}")
        print("")

    output = pd.DataFrame(results)

    output.sort_values("Total_IPG_Publications", ascending=False, inplace=True)

    output.to_csv("national_ipg_stats.csv", index=False)

    print("\nFinished.")
    print("Saved to national_ipg_stats.csv\n")


if __name__ == "__main__":
    main()
