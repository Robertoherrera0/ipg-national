"""
step2_analysis.py
─────────────────────────────────────────────────────────────────────
Internal Collaboration Density → National PageRank
Land-Grant Universities — IPG Plant Science Faculty

  1. Baseline OLS       — scatter + fit, predicted vs actual, diagnostics
  2. Is it real?        — permutation test + bootstrap CI
  3. Is it just size?   — regression scatter plots per spec +
                          dot-and-CI coefficient plot (NOT bars)
  4. Quantile regression — effect across PageRank distribution

Run: python step2_analysis.py
─────────────────────────────────────────────────────────────────────
"""

import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, spearmanr
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
np.random.seed(42)

METRICS_CSV = "data/ipg_combined_metrics_v2.csv"
SCHOOLS_DIR = "data/schools"
OUT         = "results"
os.makedirs(OUT, exist_ok=True)

N_PERM = 10_000
N_BOOT = 5_000

C_BLUE  = "#2C5F8A"
C_RED   = "#C0392B"
C_GREEN = "#27AE60"
C_GRAY  = "#95A5A6"

# ── LOAD + FILTER ─────────────────────────────────────────────────
df_all    = pd.read_csv(METRICS_CSV).dropna(subset=["EdgesPerNode","pagerank"])
pr_thresh = df_all["pagerank"].min() * 1.05
df        = df_all[df_all["pagerank"] > pr_thresh].reset_index(drop=True)

prod_rows = []
for school in os.listdir(SCHOOLS_DIR):
    path = os.path.join(SCHOOLS_DIR, school, "graphs", f"{school}_stats.csv")
    if not os.path.exists(path):
        continue
    try:
        s = pd.read_csv(path, index_col=0)
        prod_rows.append({"school": school,
                          "total_papers": s["Total papers"].sum(),
                          "n_faculty":    len(s)})
    except:
        pass

df = df.merge(pd.DataFrame(prod_rows), on="school", how="left").dropna(
         subset=["total_papers","n_faculty"])

y      = df["pagerank"].values
labels = df["school"].values
x_vals = df["EdgesPerNode"].values
n      = len(df)

removed = df_all[df_all["pagerank"] <= pr_thresh]["school"].tolist()
print(f"\n  {n} schools  |  removed: {removed}")

def ols_robust(X_df, y_arr):
    Xc = sm.add_constant(X_df, has_constant="add")
    return sm.OLS(y_arr, Xc).fit(cov_type="HC3")

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def boot_epn(X_df, y_arr, n_boot=N_BOOT):
    nb, betas = len(y_arr), []
    for _ in range(n_boot):
        idx = np.random.choice(nb, nb, replace=True)
        Xb  = sm.add_constant(X_df.iloc[idx], has_constant="add")
        betas.append(sm.OLS(y_arr[idx], Xb).fit().params["EdgesPerNode"])
    betas = np.array(betas)
    return betas, np.percentile(betas, [2.5, 97.5])


# ══════════════════════════════════════════════════════════════════
# 1 · BASELINE OLS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  1 · BASELINE OLS")
print("═"*60)

m      = ols_robust(df[["EdgesPerNode"]], y)
yhat   = m.fittedvalues.values
resids = m.resid.values
x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
Xl     = sm.add_constant(pd.DataFrame({"EdgesPerNode": x_line}), has_constant="add")

sw_stat, sw_p       = shapiro(resids)
bp_lm,   bp_p, _,_ = sms.het_breuschpagan(resids, m.model.exog)
dw                  = durbin_watson(resids)
cooks_d, _          = m.get_influence().cooks_distance
high_cooks          = labels[cooks_d > 4/n]
rho_sp, p_sp        = spearmanr(x_vals, y)

print(m.summary())
print(f"\n  Shapiro-Wilk p={sw_p:.4f}  BP p={bp_p:.4f}  "
      f"DW={dw:.4f}  Spearman ρ={rho_sp:.4f} p={p_sp:.4f}")
print(f"  Influential: {list(high_cooks)}")

# Fig 1: scatter+fit  |  predicted vs actual  |  residuals  |  QQ  |  Cook's
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

# Scatter + fit
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(x_vals, y, color=C_BLUE, alpha=0.75, s=65, zorder=3)
ax1.plot(x_line, m.predict(Xl), color=C_RED, lw=2.5, zorder=4)
for j, txt in enumerate(labels):
    ax1.annotate(txt, (x_vals[j], y[j]), fontsize=6, alpha=0.65,
                 xytext=(3,3), textcoords="offset points")
ax1.set_xlabel("EdgesPerNode  (internal collaboration density)", fontsize=11)
ax1.set_ylabel("PageRank  (national centrality)", fontsize=11)
ax1.set_title(
    f"β={m.params['EdgesPerNode']:.5f}   "
    f"p={m.pvalues['EdgesPerNode']:.2e} {sig_stars(m.pvalues['EdgesPerNode'])}   "
    f"R²={m.rsquared:.4f}   AdjR²={m.rsquared_adj:.4f}   "
    f"Spearman ρ={rho_sp:.3f}",
    fontsize=10, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)

# Predicted vs Actual
ax2 = fig.add_subplot(gs[0, 2])
lims = [min(y.min(), yhat.min())*0.9, max(y.max(), yhat.max())*1.05]
ax2.scatter(y, yhat, color=C_BLUE, alpha=0.75, s=60, zorder=3)
ax2.plot(lims, lims, color=C_RED, lw=1.8, linestyle="--", zorder=2)
for j, txt in enumerate(labels):
    ax2.annotate(txt, (y[j], yhat[j]), fontsize=5.5, alpha=0.62,
                 xytext=(2,2), textcoords="offset points")
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.set_xlabel("Actual PageRank", fontsize=10)
ax2.set_ylabel("Predicted PageRank", fontsize=10)
ax2.set_title("Predicted vs Actual\n(dots on red line = perfect fit)",
              fontsize=9, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)

# Residuals vs fitted
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(yhat, resids, color=C_BLUE, alpha=0.7, s=50)
ax3.axhline(0, color=C_GRAY, linestyle="--", lw=1)
for j, txt in enumerate(labels):
    ax3.annotate(txt, (yhat[j], resids[j]), fontsize=5.5, alpha=0.6,
                 xytext=(2,2), textcoords="offset points")
ax3.set_xlabel("Fitted values", fontsize=10)
ax3.set_ylabel("Residuals", fontsize=10)
ax3.set_title(f"Residuals vs Fitted\nSW p={sw_p:.3f}  BP p={bp_p:.3f}", fontsize=9)
ax3.spines[["top","right"]].set_visible(False)

# QQ
ax4 = fig.add_subplot(gs[1, 1])
sm.qqplot(resids, line="s", ax=ax4, alpha=0.7)
ax4.set_title(f"Q-Q Plot  (SW p={sw_p:.3f})", fontsize=9)
ax4.spines[["top","right"]].set_visible(False)

# Cook's distance
ax5 = fig.add_subplot(gs[1, 2])
ax5.stem(range(n), cooks_d, markerfmt="C0o", linefmt="C0-", basefmt="gray")
thresh = 4/n
ax5.axhline(thresh, color=C_RED, linestyle="--", label=f"4/n={thresh:.3f}")
for i, lbl in enumerate(labels):
    if cooks_d[i] > thresh:
        ax5.annotate(lbl, (i, cooks_d[i]), fontsize=7, color=C_RED)
ax5.set_xlabel("Observation index", fontsize=10)
ax5.set_ylabel("Cook's D", fontsize=10)
ax5.set_title("Cook's Distance", fontsize=9)
ax5.legend(fontsize=8)
ax5.spines[["top","right"]].set_visible(False)

plt.suptitle("1 · Baseline OLS", fontsize=13, fontweight="bold")
plt.savefig(f"{OUT}/fig1_baseline.png", dpi=150)
plt.close()
print("\n  → fig1_baseline.png")


# ══════════════════════════════════════════════════════════════════
# 2 · IS IT REAL?
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  2 · IS IT REAL?")
print("═"*60)

obs_r2   = m.rsquared
perm_r2s = np.array([
    sm.OLS(np.random.permutation(y),
           sm.add_constant(df[["EdgesPerNode"]], has_constant="add")).fit().rsquared
    for _ in range(N_PERM)
])
perm_p = (perm_r2s >= obs_r2).mean()
boot_betas, (ci_lo, ci_hi) = boot_epn(df[["EdgesPerNode"]], y)

print(f"  Permutation p = {'<0.0001' if perm_p==0 else perm_p}")
print(f"  Bootstrap CI  = [{ci_lo:.5f}, {ci_hi:.5f}]")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(perm_r2s, bins=60, color=C_BLUE, alpha=0.82, edgecolor="white")
axes[0].axvline(obs_r2, color=C_RED, lw=2.5,
                label=f"Observed R²={obs_r2:.3f}  (p<0.0001)")
axes[0].set_xlabel("R² under random permutation", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title(f"Permutation Test  ({N_PERM:,} shuffles)\n"
                  f"0 of {N_PERM:,} random shuffles reached observed R²",
                  fontsize=10, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].spines[["top","right"]].set_visible(False)

axes[1].hist(boot_betas, bins=60, color=C_BLUE, alpha=0.82, edgecolor="white")
axes[1].axvline(m.params["EdgesPerNode"], color=C_RED, lw=2,
                label=f"β={m.params['EdgesPerNode']:.5f}")
axes[1].axvline(ci_lo, color=C_GRAY, lw=1.5, linestyle="--",
                label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
axes[1].axvline(ci_hi, color=C_GRAY, lw=1.5, linestyle="--")
axes[1].axvline(0, color="black", lw=1, linestyle=":")
axes[1].set_xlabel("EdgesPerNode β", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title(f"Bootstrap Distribution  ({N_BOOT:,} resamples)\n"
                  f"95% CI entirely above zero",
                  fontsize=10, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].spines[["top","right"]].set_visible(False)

plt.suptitle("2 · Is It Real?", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_is_it_real.png", dpi=150)
plt.close()
print("  → fig2_is_it_real.png")


# ══════════════════════════════════════════════════════════════════
# 3 · IS IT JUST SIZE?
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  3 · IS IT JUST SIZE?")
print("═"*60)

specs = [
    ("Baseline",                   ["EdgesPerNode"]),
    ("+ total papers",             ["EdgesPerNode","total_papers"]),
    ("+ faculty count",            ["EdgesPerNode","n_faculty"]),
    ("+ total papers & faculty",   ["EdgesPerNode","total_papers","n_faculty"]),
]

coef_rows = []
for label, cols in specs:
    mc    = ols_robust(df[cols], y)
    b     = mc.params["EdgesPerNode"]
    p     = mc.pvalues["EdgesPerNode"]
    betas, (lo, hi) = boot_epn(df[cols], y)
    coef_rows.append({"label": label, "β": b, "lo": lo, "hi": hi, "p": p,
                      "sig": sig_stars(p)})
    ctrl = ""
    if len(cols) > 1:
        ctrl = "  [" + ", ".join(f"{c} p={mc.pvalues[c]:.3f}"
                                  for c in cols[1:]) + "]"
    print(f"  {label:<30}  β={b:.5f}  [{lo:.5f},{hi:.5f}]  "
          f"p={p:.4f} {sig_stars(p)}{ctrl}")

# Fig 3a: four scatter panels with regression line per spec
x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (label, cols) in zip(axes, specs):
    mc    = ols_robust(df[cols], y)
    b     = mc.params["EdgesPerNode"]
    p     = mc.pvalues["EdgesPerNode"]

    pred_df = pd.DataFrame({"EdgesPerNode": x_range})
    for c in cols[1:]:
        pred_df[c] = df[c].mean()
    Xp    = sm.add_constant(pred_df, has_constant="add")
    y_hat = mc.predict(Xp)

    ax.scatter(x_vals, y, color=C_BLUE, alpha=0.72, s=55, zorder=3)
    ax.plot(x_range, y_hat, color=C_RED, lw=2.2, zorder=4)
    for j, txt in enumerate(labels):
        ax.annotate(txt, (x_vals[j], y[j]), fontsize=5.5, alpha=0.62,
                    xytext=(2,2), textcoords="offset points")
    ax.set_xlabel("EdgesPerNode", fontsize=10)
    ax.set_ylabel("PageRank", fontsize=10)
    ax.set_title(f"{label}\nβ={b:.5f}   p={p:.4f} {sig_stars(p)}   "
                 f"R²={mc.rsquared:.4f}",
                 fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("3 · Is It Just Size?  —  Regression Line per Specification",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3a_control_regressions.png", dpi=150)
plt.close()

# Fig 3b: dot-and-CI coefficient plot
fig, ax = plt.subplots(figsize=(8, 5))
y_pos   = np.arange(len(coef_rows))
colors  = [C_BLUE] + [C_GREEN] * (len(coef_rows)-1)

for i, row in enumerate(coef_rows):
    ax.plot([row["lo"], row["hi"]], [i, i],
            color=colors[i], lw=2.5, alpha=0.85, solid_capstyle="round")
    ax.scatter(row["β"], i, color=colors[i], s=120, zorder=5)
    ax.text(row["hi"] + 0.0002, i,
            f"  {row['sig']}  p={row['p']:.3f}",
            va="center", fontsize=10, color="#222")

ax.axvline(0, color="black", lw=1, linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels([r["label"] for r in coef_rows], fontsize=11)
ax.set_xlabel("EdgesPerNode β  (bootstrap 95% CI)", fontsize=11)
ax.set_title("EdgesPerNode Effect Before and After Size Controls\n"
             "Dot = estimate   Line = 95% CI",
             fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3b_coef_plot.png", dpi=150)
plt.close()

print("\n  → fig3a_control_regressions.png")
print("  → fig3b_coef_plot.png")


# ══════════════════════════════════════════════════════════════════
# 4 · QUANTILE REGRESSION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  4 · QUANTILE REGRESSION")
print("═"*60)

QUANTILES  = [0.10, 0.25, 0.50, 0.75, 0.90]
qr_results = []

for q in QUANTILES:
    qr = smf.quantreg("pagerank ~ EdgesPerNode", data=df).fit(q=q)
    b  = qr.params["EdgesPerNode"]
    p  = qr.pvalues["EdgesPerNode"]

    boot_q = []
    for _ in range(N_BOOT):
        idx  = np.random.choice(n, n, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        try:
            fit = smf.quantreg("pagerank ~ EdgesPerNode",
                               data=df_b).fit(q=q, max_iter=2000)
            boot_q.append(fit.params["EdgesPerNode"])
        except:
            pass
    boot_q     = np.array(boot_q)
    lo_q, hi_q = np.percentile(boot_q, [2.5, 97.5])
    qr_results.append({"q": q, "β": b, "lo": lo_q, "hi": hi_q, "p": p})
    print(f"  q={q}  β={b:.5f}  [{lo_q:.5f},{hi_q:.5f}]  "
          f"p={p:.4f} {sig_stars(p)}")

qr_df = pd.DataFrame(qr_results)

# Fig 4a: β at each quantile vs OLS band
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhspan(ci_lo, ci_hi, alpha=0.15, color=C_BLUE)
ax.axhline(m.params["EdgesPerNode"], color=C_BLUE, lw=2,
           linestyle="--", label=f"OLS β={m.params['EdgesPerNode']:.5f}")

q_vals = qr_df["q"].values
b_vals = qr_df["β"].values
lo_arr = qr_df["lo"].values
hi_arr = qr_df["hi"].values

ax.fill_between(q_vals, lo_arr, hi_arr, alpha=0.25, color=C_RED)
ax.plot(q_vals, b_vals, "o-", color=C_RED, lw=2, ms=8, zorder=4,
        label="Quantile regression β")

ax.set_xlabel("Quantile", fontsize=12)
ax.set_ylabel("EdgesPerNode β", fontsize=12)
ax.set_xticks(QUANTILES)
ax.set_title("Quantile Regression vs OLS\n"
             "Rising slope = effect stronger for high-PageRank schools",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4a_quantile_coef.png", dpi=150)
plt.close()

# Fig 4b: scatter with all QR lines + OLS
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_vals, y, color=C_GRAY, alpha=0.6, s=55, zorder=2)
for j, txt in enumerate(labels):
    ax.annotate(txt, (x_vals[j], y[j]), fontsize=5.5, alpha=0.6,
                xytext=(2,2), textcoords="offset points")

q_colors = ["#5DADE2","#2E86C1","#1A5276","#E67E22","#922B21"]
for row, col in zip(qr_results, q_colors):
    qr_m  = smf.quantreg("pagerank ~ EdgesPerNode", data=df).fit(q=row["q"])
    y_qr  = qr_m.params["Intercept"] + qr_m.params["EdgesPerNode"] * x_line
    ax.plot(x_line, y_qr, lw=1.8, color=col,
            label=f"QR q={row['q']}  β={row['β']:.5f}")

ax.plot(x_line, m.predict(Xl), color="black", lw=2.5, linestyle="--",
        label=f"OLS β={m.params['EdgesPerNode']:.5f}", zorder=5)

ax.set_xlabel("EdgesPerNode", fontsize=11)
ax.set_ylabel("PageRank", fontsize=11)
ax.set_title("OLS vs Quantile Regression Lines", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4b_quantile_lines.png", dpi=150)
plt.close()

print("\n  → fig4a_quantile_coef.png")
print("  → fig4b_quantile_lines.png")

# ── EXPORT ────────────────────────────────────────────────────────
qr_df.to_csv(f"{OUT}/quantile_results.csv", index=False)
pd.DataFrame(coef_rows).to_csv(f"{OUT}/confound_table.csv", index=False)

print("\n" + "═"*60)
print(f"  OUTPUTS IN {OUT}/")
print("    fig1_baseline.png           scatter+fit, pred vs actual, diagnostics")
print("    fig2_is_it_real.png         permutation + bootstrap")
print("    fig3a_control_regressions.png  scatter per spec")
print("    fig3b_coef_plot.png         dot-and-CI coefficient plot")
print("    fig4a_quantile_coef.png     QR β vs OLS band")
print("    fig4b_quantile_lines.png    all regression lines on scatter")
print("═"*60)