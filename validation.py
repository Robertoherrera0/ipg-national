import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("data/ipg_all_metrics.csv")
df = df.dropna(subset=['EdgesPerNode', 'sigma', 'pagerank'])
df = df[df['pagerank'] > 0]

df['EdgesPerNode_log'] = np.log1p(df['EdgesPerNode'])
df['sigma_log'] = np.log1p(df['sigma'])

model = smf.glm("pagerank ~ EdgesPerNode_log + sigma_log", data=df,
               family=sm.families.Gamma(link=sm.families.links.Log())).fit()

df['predicted_pagerank'] = model.fittedvalues
df['residual'] = df['pagerank'] - df['predicted_pagerank']
df['pct_error'] = (df['residual'] / df['predicted_pagerank']) * 100

print("="*90)
print("OVERPERFORMERS vs UNDERPERFORMERS")
print("="*90)

print("\nTOP 10 OVERPERFORMERS (actual > predicted):")
top_over = df.nlargest(10, 'residual')[['school', 'pagerank', 'predicted_pagerank', 'residual', 'pct_error']]
for idx, row in top_over.iterrows():
    print(f"{row['school']:<20s}: actual={row['pagerank']:.5f}, predicted={row['predicted_pagerank']:.5f}, +{row['pct_error']:.1f}%")

print("\nTOP 10 UNDERPERFORMERS (actual < predicted):")
top_under = df.nsmallest(10, 'residual')[['school', 'pagerank', 'predicted_pagerank', 'residual', 'pct_error']]
for idx, row in top_under.iterrows():
    print(f"{row['school']:<20s}: actual={row['pagerank']:.5f}, predicted={row['predicted_pagerank']:.5f}, {row['pct_error']:.1f}%")

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(df['predicted_pagerank'], df['pagerank'], alpha=0.6, s=120, 
           edgecolor='black', color='steelblue', linewidth=2)
plt.plot([df['predicted_pagerank'].min(), df['predicted_pagerank'].max()],
         [df['predicted_pagerank'].min(), df['predicted_pagerank'].max()],
         'r--', linewidth=2, label='Perfect prediction')

# Label outliers
for idx, row in df.iterrows():
    if abs(row['pct_error']) > 50:
        plt.annotate(row['school'], (row['predicted_pagerank'], row['pagerank']),
                    fontsize=9, alpha=0.7)

plt.xlabel('Predicted PageRank', fontsize=13, fontweight='bold')
plt.ylabel('Actual PageRank', fontsize=13, fontweight='bold')
plt.title('Overperformers vs Underperformers', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*90)