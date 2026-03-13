import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

DATA="data/ipg_metrics_with_pagerank.csv"
OUT="analysis_outputs"
os.makedirs(OUT,exist_ok=True)

predictors=[
"EdgesPerNode",
"assortativity_unweighted",
"global_efficiency",
"local_efficiency",
"avg_path_over_diameter",
"clique_integration",
"sigma_small_world_index",
"omega_small_world_index"
]

df=pd.read_csv(DATA)

X=df[predictors]
y=df["pagerank"]

pearson=df[predictors+["pagerank"]].corr(method="pearson")
spearman=df[predictors+["pagerank"]].corr(method="spearman")

Xc=sm.add_constant(X)
ols=sm.OLS(y,Xc).fit()
robust=ols.get_robustcov_results(cov_type="HC3")

scaler=StandardScaler()
Xs=pd.DataFrame(scaler.fit_transform(X),columns=predictors)
ys=(y-y.mean())/y.std()
std_model=sm.OLS(ys,sm.add_constant(Xs)).fit()
std_coefs=std_model.params[1:]

lasso=LassoCV(cv=5,max_iter=10000).fit(Xs,ys)
lasso_coefs=pd.Series(lasso.coef_,index=predictors)

vif=pd.DataFrame()
vif["variable"]=predictors
vif["VIF"]=[variance_inflation_factor(X.values,i) for i in range(len(predictors))]

ranking=df.sort_values("pagerank",ascending=False)

coef_table=pd.DataFrame({
"coef":ols.params,
"p_value":ols.pvalues
})

coef_robust=pd.DataFrame({
"coef":robust.params,
"p_value_HC3":robust.pvalues
})

std_table=pd.DataFrame({
"standardized_beta":std_coefs
})

lasso_table=pd.DataFrame({
"lasso_coef":lasso_coefs
})

excel=f"{OUT}/pagerank_analysis.xlsx"

with pd.ExcelWriter(excel) as w:
    coef_table.to_excel(w,"OLS")
    coef_robust.to_excel(w,"Robust_HC3")
    std_table.to_excel(w,"Standardized_Betas")
    lasso_table.to_excel(w,"LASSO")
    vif.to_excel(w,"VIF")
    pearson.to_excel(w,"Pearson_Corr")
    spearman.to_excel(w,"Spearman_Corr")
    ranking.to_excel(w,"PageRank_Ranking")

plt.figure(figsize=(12,10))
sns.heatmap(pearson,annot=True,cmap="coolwarm",vmin=-1,vmax=1,square=True)
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png",dpi=600)
plt.close()

for v in predictors:
    plt.figure(figsize=(8,6))
    sns.regplot(data=df,x=v,y="pagerank",scatter_kws={"s":80},line_kws={"color":"red"})
    plt.tight_layout()
    plt.savefig(f"{OUT}/regression_{v}.png",dpi=600)
    plt.close()

plt.figure(figsize=(10,12))
sns.barplot(data=ranking,y="school",x="pagerank",color="steelblue")
plt.tight_layout()
plt.savefig(f"{OUT}/pagerank_ranking.png",dpi=600)
plt.close()

plt.figure(figsize=(8,6))
sns.histplot(df["pagerank"],kde=True)
plt.tight_layout()
plt.savefig(f"{OUT}/pagerank_distribution.png",dpi=600)
plt.close()

print("\nOLS\n")
print(ols.summary())

print("\nROBUST HC3\n")
print(robust.summary())

print("\nSTANDARDIZED COEFFICIENTS\n")
print(std_coefs.sort_values(key=abs,ascending=False))

print("\nLASSO COEFFICIENTS\n")
print(lasso_coefs.sort_values(key=abs,ascending=False))

print("\nVIF\n")
print(vif.sort_values("VIF",ascending=False))

print("\nOutput saved to:",OUT)