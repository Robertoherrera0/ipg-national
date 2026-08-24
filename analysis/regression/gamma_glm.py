import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_FILE = "data/ipg_metrics.csv"
OUTPUT_DIR = "results/tables"
RESPONSE = "pagerank"

PREDICTORS = [
    "EdgesPerNode",
    "clique_integration",
    "global_efficiency",
    "local_efficiency",
]


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=PREDICTORS + [RESPONSE])
    df = df[df[RESPONSE] > 0].reset_index(drop=True)
    return df


def fit_gamma_glm(formula, data):
    return smf.glm(formula, data=data,
                   family=sm.families.Gamma(link=sm.families.links.Log())).fit()


def pseudo_r2(model):
    return 1 - model.deviance / model.null_deviance


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def run_univariate(df):
    print("=" * 90)
    print(f"UNIVARIATE MODELS: each predictor -> {RESPONSE}")
    print("=" * 90)
    print(f"\n{'Predictor':<25s} {'beta':>10s} {'p-value':>10s} {'Pseudo-R2':>10s} {'n':>5s}")
    print("-" * 65)
    
    rows = []
    for pred in PREDICTORS:
        data = df[[pred, RESPONSE]].dropna()
        model = fit_gamma_glm(f"{RESPONSE} ~ {pred}", data)
        r2 = pseudo_r2(model)
        rows.append({
            "Predictor": pred,
            "beta": model.params[pred],
            "p_value": model.pvalues[pred],
            "Pseudo_R2": r2,
            "n": len(data),
            "significance": sig_stars(model.pvalues[pred])
        })
        print(f"{pred:<25s} {model.params[pred]:>10.4f} {model.pvalues[pred]:>10.4f} "
              f"{sig_stars(model.pvalues[pred]):<3s}{r2:>7.3f} {len(data):>5d}")
    
    univariate_df = pd.DataFrame(rows)
    univariate_df.to_csv(os.path.join(OUTPUT_DIR, f"{RESPONSE}_univariate.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/{RESPONSE}_univariate.csv")


def run_full_model(df):
    print("\n" + "=" * 90)
    print(f"FULL MODEL: all predictors -> {RESPONSE}")
    print("=" * 90)
    formula = f"{RESPONSE} ~ " + " + ".join(PREDICTORS)
    data = df[PREDICTORS + [RESPONSE]].dropna()
    model = fit_gamma_glm(formula, data)
    r2 = pseudo_r2(model)
    print(f"\nSample size: {len(data)}  Pseudo-R2: {r2:.3f}  AIC: {model.aic:.2f}")
    print(f"\n{'Predictor':<25s} {'beta':>10s} {'SE':>10s} {'p-value':>10s} {'95% CI':>22s}")
    print("-" * 85)
    
    rows = []
    for pred in PREDICTORS:
        ci = model.conf_int().loc[pred]
        rows.append({
            "Predictor": pred,
            "beta": model.params[pred],
            "SE": model.bse[pred],
            "p_value": model.pvalues[pred],
            "CI_lower": ci[0],
            "CI_upper": ci[1],
            "significance": sig_stars(model.pvalues[pred])
        })
        print(f"{pred:<25s} {model.params[pred]:>10.4f} {model.bse[pred]:>10.4f} "
              f"{model.pvalues[pred]:>10.4f} {sig_stars(model.pvalues[pred]):<3s}"
              f" [{ci[0]:>7.4f}, {ci[1]:>7.4f}]")
    
    full_model_df = pd.DataFrame(rows)
    full_model_df["Pseudo_R2"] = r2
    full_model_df["n"] = len(data)
    full_model_df["AIC"] = model.aic
    full_model_df.to_csv(os.path.join(OUTPUT_DIR, f"{RESPONSE}_full_model.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/{RESPONSE}_full_model.csv")
    
    return data, r2


def run_leave_one_out(data, full_r2):
    print("\n" + "=" * 90)
    print("LEAVE-ONE-OUT: unique contribution of each predictor")
    print("=" * 90)
    print(f"\nBaseline Pseudo-R2: {full_r2:.3f}")
    print(f"\n{'Dropped Predictor':<25s} {'R2 without':>12s} {'R2 drop':>10s} {'% drop':>8s}")
    print("-" * 60)
    
    rows = []
    for pred in PREDICTORS:
        remaining = [p for p in PREDICTORS if p != pred]
        model = fit_gamma_glm(f"{RESPONSE} ~ " + " + ".join(remaining), data)
        r2 = pseudo_r2(model)
        drop = full_r2 - r2
        pct = (drop / full_r2) * 100
        rows.append({
            "Dropped_Predictor": pred,
            "R2_without": r2,
            "R2_drop": drop,
            "Percent_drop": pct
        })
        print(f"{pred:<25s} {r2:>12.3f} {drop:>10.3f} {pct:>7.1f}%")
    
    leave_one_out_df = pd.DataFrame(rows)
    leave_one_out_df.to_csv(os.path.join(OUTPUT_DIR, f"{RESPONSE}_leave_one_out.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/{RESPONSE}_leave_one_out.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} schools from {DATA_FILE}  |  RESPONSE = {RESPONSE}\n")
    run_univariate(df)
    data, r2 = run_full_model(df)
    run_leave_one_out(data, r2)


if __name__ == "__main__":
    main()