import pandas as pd

absolute_csv = "data/national/national_ipg_matrix.csv"
relative_csv = "data/national/national_ipg_matrix_relative.csv"
unweighted_csv = "data/national/national_ipg_matrix_unweighted.csv"

def main():

    absolute = pd.read_csv(absolute_csv, index_col=0)
    schools = absolute.index.tolist()

    relative = pd.DataFrame(0.0, index=schools, columns=schools)

    for school in schools:

        external_total = absolute.loc[school].drop(school).sum()

        if external_total == 0:
            continue

        for other in schools:
            if other == school:
                continue

            weight = absolute.loc[school, other]

            if weight > 0:
                relative.loc[school, other] = weight / external_total

    relative.to_csv(relative_csv)

    unweighted = pd.DataFrame(0, index=schools, columns=schools)

    for i in schools:
        for j in schools:
            if i != j and absolute.loc[i, j] > 0:
                unweighted.loc[i, j] = 1

    unweighted.to_csv(unweighted_csv)

    print("Done.")

if __name__ == "__main__":
    main()
