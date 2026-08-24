import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson

DATA_FILE = "data/ipg_metrics.csv"
OUTPUT_DIR = "results/tables"
OUTCOME = "pagerank"

PREDICTORS = [
    "EdgesPerNode",
    "clique_integration",
    "global_efficiency",
    "local_efficiency",
]


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=PREDICTORS + [OUTCOME])
    df = df[df[OUTCOME] > 0].reset_index(drop=True)
    return df


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def check_outcome_distribution(df):
    print("=" * 90)
    print(f"OUTCOME DISTRIBUTION: {OUTCOME}")
    print("=" * 90)
    
    y = df[OUTCOME]
    sw_stat, sw_p = shapiro(y)
    skewness = pd.Series(y).skew()
    
    print(f"\nShapiro-Wilk (normality): stat={sw_stat:.4f}  p={sw_p:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Min: {y.min():.6f}  Max: {y.max():.6f}  Mean: {y.mean():.6f}")
    
    if sw_p < 0.05:
        print(f"\nOutcome violates normality (p<0.05). Right-skewed positive data.")
        print("→ Gamma GLM (log link) is appropriate.\n")
        return "gamma"
    else:
        print(f"\nOutcome approximately normal (p>=0.05).")
        print("→ OLS is appropriate.\n")
        return "ols"


def fit_ols(df):
    X = sm.add_constant(df[PREDICTORS])
    return sm.OLS(df[OUTCOME], X).fit()


def fit_gamma_glm(formula, data):
    return smf.glm(formula, data=data,
                   family=sm.families.Gamma(link=sm.families.links.Log())).fit()


def pseudo_r2(model):
    return 1 - model.deviance / model.null_deviance


def run_univariate(df):
    print("=" * 90)
    print(f"UNIVARIATE MODELS: each predictor -> {OUTCOME}")
    print("=" * 90)
    print(f"\n{'Predictor':<25s} {'beta':>10s} {'p-value':>10s} {'Pseudo-R2':>10s} {'n':>5s}")
    print("-" * 65)

    rows = []
    for pred in PREDICTORS:
        data = df[[pred, OUTCOME]].dropna()
        model = fit_gamma_glm(f"{OUTCOME} ~ {pred}", data)
        beta = model.params[pred]
        pval = model.pvalues[pred]
        r2 = pseudo_r2(model)
        rows.append({
            "Predictor": pred,
            "beta": beta,
            "p_value": pval,
            "Pseudo_R2": r2,
            "n": len(data),
            "significance": sig_stars(pval)
        })
        print(f"{pred:<25s} {beta:>10.4f} {pval:>10.4f} "
              f"{sig_stars(pval):<3s}{r2:>7.3f} {len(data):>5d}")

    univariate_df = pd.DataFrame(rows)
    univariate_df.to_csv(os.path.join(OUTPUT_DIR, f"{OUTCOME}_univariate.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/{OUTCOME}_univariate.csv")


def run_full_model(df):
    print("\n" + "=" * 90)
    print(f"FULL MODEL: all predictors -> {OUTCOME}")
    print("=" * 90)

    formula = f"{OUTCOME} ~ " + " + ".join(PREDICTORS)
    data = df[PREDICTORS + [OUTCOME]].dropna()
    model = fit_gamma_glm(formula, data)
    r2 = pseudo_r2(model)

    print(f"\nSample size: {len(data)}  Pseudo-R2: {r2:.3f}  AIC: {model.aic:.2f}")
    print(f"\n{'Predictor':<25s} {'beta':>10s} {'SE':>10s} {'p-value':>10s} {'95% CI':>22s}")
    print("-" * 85)

    rows = []
    for pred in PREDICTORS:
        beta = model.params[pred]
        se = model.bse[pred]
        pval = model.pvalues[pred]
        ci = model.conf_int().loc[pred]
        rows.append({
            "Predictor": pred,
            "beta": beta,
            "SE": se,
            "p_value": pval,
            "CI_lower": ci[0],
            "CI_upper": ci[1],
            "significance": sig_stars(pval)
        })
        print(f"{pred:<25s} {beta:>10.4f} {se:>10.4f} "
              f"{pval:>10.4f} {sig_stars(pval):<3s}"
              f" [{ci[0]:>7.4f}, {ci[1]:>7.4f}]")

    full_model_df = pd.DataFrame(rows)
    full_model_df["Pseudo_R2"] = r2
    full_model_df["n"] = len(data)
    full_model_df["AIC"] = model.aic
    full_model_df.to_csv(os.path.join(OUTPUT_DIR, f"{OUTCOME}_full_model.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/{OUTCOME}_full_model.csv")

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
        model = fit_gamma_glm(f"{OUTCOME} ~ " + " + ".join(remaining), data)
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
    leave_one_out_df.to_csv(os.path.join(OUTPUT_DIR, f"{OUTCOME}_leave_one_out.csv"), index=False)
    print(f"\nSaved: {OUTPUT_DIR}/{OUTCOME}_leave_one_out.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} schools from {DATA_FILE}\n")

    model_choice = check_outcome_distribution(df)
    
    print("=" * 90)
    print("FINAL MODEL RESULTS (Gamma GLM)")
    print("=" * 90)

    run_univariate(df)
    data, r2 = run_full_model(df)
    run_leave_one_out(data, r2)


if __name__ == "__main__":
    main()