import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_FILE = "data/ipg_metrics.csv"
OUTPUT_DIR = "results/tables"

PREDICTORS = ["EdgesPerNode", "clique_integration", "global_efficiency", "local_efficiency"]

RESPONSES = [
    "degree_centrality",
    "eigenvector_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "clustering",
]


def load_data():
    df = pd.read_csv(DATA_FILE)
    return df


def check_outcome_normality(y):
    sw_stat, sw_p = shapiro(y)
    return sw_p


def fit_gamma_glm(formula, data):
    return smf.glm(formula, data=data,
                   family=sm.families.Gamma(link=sm.families.links.Log())).fit()


def fit_ols(formula, data):
    return smf.ols(formula, data=data).fit()


def pseudo_r2(model):
    return 1 - model.deviance / model.null_deviance


def r2(model):
    return model.rsquared


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def run_pairwise(df):
    print("=" * 100)
    print("PAIRWISE: each internal predictor -> each external outcome")
    print("=" * 100)

    all_rows = []
    for outcome in RESPONSES:
        data = df[[outcome]].dropna()
        sw_p = check_outcome_normality(data[outcome])
        model_type = "Gamma GLM" if sw_p < 0.05 else "OLS"
        
        print(f"\n--- Outcome: {outcome} ({model_type}, Shapiro-Wilk p={sw_p:.4f}) ---")
        print(f"{'Predictor':<25s} {'beta':>10s} {'p-value':>10s} {'R2':>10s} {'n':>5s}")
        print("-" * 65)

        for pred in PREDICTORS:
            data = df[[pred, outcome]].dropna()
            data = data[data[outcome] > 0]
            if len(data) < 5:
                print(f"{pred:<25s} {'skipped (n<5)':>10s}")
                continue

            if sw_p < 0.05:
                model = fit_gamma_glm(f"{outcome} ~ {pred}", data)
                model_r2 = pseudo_r2(model)
            else:
                model = fit_ols(f"{outcome} ~ {pred}", data)
                model_r2 = r2(model)

            all_rows.append({
                "Outcome": outcome,
                "Model": model_type,
                "Shapiro_p": sw_p,
                "Predictor": pred,
                "beta": model.params[pred],
                "p_value": model.pvalues[pred],
                "R2": model_r2,
                "n": len(data),
                "significance": sig_stars(model.pvalues[pred])
            })
            print(f"{pred:<25s} {model.params[pred]:>10.4f} {model.pvalues[pred]:>10.4f} "
                  f"{sig_stars(model.pvalues[pred]):<3s}{model_r2:>7.3f} {len(data):>5d}")

    pairwise_df = pd.DataFrame(all_rows)
    pairwise_df.to_csv(os.path.join(OUTPUT_DIR, "intra_inter_pairwise.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/intra_inter_pairwise.csv")


def run_full_model_per_outcome(df):
    print("\n" + "=" * 100)
    print("FULL MODEL: all internal predictors -> each external outcome")
    print("=" * 100)

    formula = " + ".join(PREDICTORS)
    all_rows = []
    
    for outcome in RESPONSES:
        data = df[PREDICTORS + [outcome]].dropna()
        data = data[data[outcome] > 0]
        if len(data) < 10:
            print(f"{outcome:<25s} skipped (n<10)")
            continue

        sw_p = check_outcome_normality(data[outcome])
        model_type = "Gamma GLM" if sw_p < 0.05 else "OLS"

        if sw_p < 0.05:
            model = fit_gamma_glm(f"{outcome} ~ {formula}", data)
            model_r2 = pseudo_r2(model)
        else:
            model = fit_ols(f"{outcome} ~ {formula}", data)
            model_r2 = r2(model)

        row = {
            "Outcome": outcome,
            "Model": model_type,
            "Shapiro_p": sw_p,
            "n": len(data),
            "R2": model_r2,
            "AIC": model.aic
        }
        for pred in PREDICTORS:
            row[f"{pred}_beta"] = model.params[pred]
            row[f"{pred}_p_value"] = model.pvalues[pred]
            row[f"{pred}_significance"] = sig_stars(model.pvalues[pred])
        all_rows.append(row)

        print(f"{outcome:<25s} {model_type:<10s} p={sw_p:.4f}  n={len(data):>3d}  R2={model_r2:.3f}")

    full_model_df = pd.DataFrame(all_rows)
    full_model_df.to_csv(os.path.join(OUTPUT_DIR, "intra_inter_full_model.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/intra_inter_full_model.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} schools from {DATA_FILE}\n")
    
    run_pairwise(df)
    run_full_model_per_outcome(df)


if __name__ == "__main__":
    main()