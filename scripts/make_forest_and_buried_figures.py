"""
Two new thesis figures:
  1) burial_forest.png     Forest plot of Table tab:adjust: Q1 patch OR and Q2 patch
                            coefficient, unadjusted vs RSA-adjusted, with 95% CIs,
                            reproducing the exact regressions behind REPORT_extended.md.
  2) buried_discordance.png Small supplementary panel showing the discordant buried-only
                            replication correlations (rho=0.17 Giacomelli vs -0.25 Kato)
                            mentioned in results.tex Section r-rep but never plotted.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(r"D:\Desktop\test 2")
FIG = ROOT / "thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1) Forest plot: Q1 (logistic OR) and Q2 (OLS coefficient), unadjusted vs RSA-adjusted
# ============================================================
exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
y_q1 = exp["destabilizing"].astype(float).values
X1_unadj = sm.add_constant(exp[["in_patch"]].astype(float).values)
X1_adj = sm.add_constant(exp[["in_patch", "rsa"]].astype(float).values)
m1u = sm.Logit(y_q1, X1_unadj).fit(disp=0)
m1a = sm.Logit(y_q1, X1_adj).fit(disp=0)

or_u, or_u_lo, or_u_hi = np.exp(m1u.params[1]), np.exp(m1u.conf_int()[1][0]), np.exp(m1u.conf_int()[1][1])
or_a, or_a_lo, or_a_hi = np.exp(m1a.params[1]), np.exp(m1a.conf_int()[1][0]), np.exp(m1a.conf_int()[1][1])

y_q2 = exp["abs_rep_ddg"].values
X2_unadj = sm.add_constant(exp[["patch_score"]].astype(float).values)
X2_adj = sm.add_constant(exp[["patch_score", "rsa"]].astype(float).values)
o1 = sm.OLS(y_q2, X2_unadj).fit()
o2 = sm.OLS(y_q2, X2_adj).fit()

c_u, c_u_lo, c_u_hi = o1.params[1], o1.conf_int()[1][0], o1.conf_int()[1][1]
c_a, c_a_lo, c_a_hi = o2.params[1], o2.conf_int()[1][0], o2.conf_int()[1][1]

print(f"Q1 OR unadj={or_u:.2f} [{or_u_lo:.2f},{or_u_hi:.2f}] p={m1u.pvalues[1]:.3g}")
print(f"Q1 OR adj  ={or_a:.2f} [{or_a_lo:.2f},{or_a_hi:.2f}] p={m1a.pvalues[1]:.3g}")
print(f"Q2 coef unadj={c_u:.4f} [{c_u_lo:.4f},{c_u_hi:.4f}] p={o1.pvalues[1]:.3g}")
print(f"Q2 coef adj  ={c_a:.4f} [{c_a_lo:.4f},{c_a_hi:.4f}] p={o2.pvalues[1]:.3g}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

ax = axes[0]
ys = [1, 0]
vals = [or_u, or_a]
los = [or_u - or_u_lo, or_a - or_a_lo]
his = [or_u_hi - or_u, or_a_hi - or_a]
colors = ["#1f4e8c", "#c0392b"]
labels = ["Unadjusted", "RSA-adjusted"]
for y, v, lo, hi, c in zip(ys, vals, los, his, colors):
    ax.errorbar(v, y, xerr=[[lo], [hi]], fmt="o", color=c, ms=9, capsize=5, lw=2)
ax.axvline(1, color="grey", ls=":", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels)
ax.set_xscale("log")
ax.set_xlabel("Q1 patch odds ratio (95% CI)")
ax.set_title("Q1: enrichment")
ax.set_ylim(-0.7, 1.7)

ax = axes[1]
vals = [c_u, c_a]
los = [c_u - c_u_lo, c_a - c_a_lo]
his = [c_u_hi - c_u, c_a_hi - c_a]
for y, v, lo, hi, c in zip(ys, vals, los, his, colors):
    ax.errorbar(v, y, xerr=[[lo], [hi]], fmt="o", color=c, ms=9, capsize=5, lw=2)
ax.axvline(0, color="grey", ls=":", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels)
ax.set_xlabel("Q2 patch-score coefficient (95% CI)")
ax.set_title("Q2: intensity")
ax.set_ylim(-0.7, 1.7)

fig.suptitle("The patch effect before and after adjustment for solvent burial (RSA)", y=1.04)
fig.tight_layout()
fig.savefig(FIG / "burial_forest.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# ============================================================
# 2) Buried-only replication discordance (Giacomelli rho=0.17 vs Kato rho=-0.25)
# ============================================================
three1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T',
'Trp':'W','Tyr':'Y','Val':'V'}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')
HOTSPOTS = [('R',175,'H'),('R',248,'W'),('R',248,'Q'),('R',249,'S'),
            ('G',245,'S'),('R',273,'H'),('R',282,'W')]
RSA_CUT = 0.10

struct = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position","patch_score","rsa"]]

def load(fname):
    df = pd.read_csv(ROOT / "functional_ext" / "data" / fname)
    rows = []
    for _, r in df.iterrows():
        m = RX.match(str(r['hgvs_pro']))
        if not m: continue
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt in three1 and mut in three1 and wt != mut and pd.notna(r['score']):
            rows.append((pos, three1[wt], three1[mut], float(r['score'])))
    return pd.DataFrame(rows, columns=['position', 'wt', 'mut', 'score'])

def orient(v):
    hs = pd.concat([v[(v.position == p) & (v.wt == w) & (v.mut == m)] for w, p, m in HOTSPOTS])
    if len(hs) == 0: return v
    if hs.score.mean() < v.score.mean():
        v = v.copy(); v["score"] = -v["score"]
    return v

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, fname, color) in zip(axes, [
        ("Giacomelli 2018", "giacomelli_scores.csv", "#1f77b4"),
        ("Kato 2003", "kato_scores.csv", "#2ca02c")]):
    v = orient(load(fname))
    res = v.groupby("position", as_index=False).agg(damage=("score", "mean"))
    d = res.merge(struct, on="position", how="inner").dropna(subset=["rsa"])
    buried = d[d.rsa < RSA_CUT]
    rho, p = spearmanr(buried.damage, buried.patch_score)
    ax.scatter(buried.patch_score, buried.damage, c=color, s=30, alpha=0.75, edgecolor="k", lw=0.3)
    if len(buried) > 2:
        z = np.polyfit(buried.patch_score, buried.damage, 1)
        xs = np.linspace(buried.patch_score.min(), buried.patch_score.max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", lw=1)
    ax.set_xlabel("aggregated 3D-patch score")
    ax.set_ylabel("functional damage score")
    ax.set_title(f"{name}, buried only (RSA$<${RSA_CUT})\nn={len(buried)}: rho={rho:.2f}, p={p:.2g}")
    print(f"{name} buried-only: n={len(buried)} rho={rho:.3f} p={p:.3g}")

fig.suptitle("Discordant buried-only intensity correlation across replication assays", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "buried_discordance.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

print("\nSaved:", FIG / "burial_forest.png")
print("Saved:", FIG / "buried_discordance.png")
