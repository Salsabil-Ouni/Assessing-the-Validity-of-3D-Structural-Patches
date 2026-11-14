"""
Extend the burial-confounder analysis (the thesis's central finding) to the
second, independent structure 2XWR, closing the "single-structure" caveat
attached to every second-structure section so far. Identical logic to
consistent_context.py / extended_analysis.py on 2OCJ, applied to
residue_table_2xwr.csv (already computed: ala_ddg, patch_score, rsa, in_patch
for all 194 shared residues).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats.contingency import odds_ratio
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]
RSA_CUT = 0.10  # matches buried_only.py's cutoff on 2OCJ

full = pd.read_csv(RES / "residue_table_2xwr.csv")  # position, ala_ddg, patch_score, rsa, is_hotspot, in_patch
A = pd.read_csv(RES / "listA_variants.csv")
rep = A.groupby("position", as_index=False).agg(rep_ddg=("ddg", "min"))
rep["destabilizing"] = rep["rep_ddg"] < DEL_BELOW
rep["abs_rep_ddg"] = rep["rep_ddg"].abs()
rep = rep.merge(full[["position", "patch_score", "rsa", "in_patch"]], on="position", how="inner")
print(f"n = {len(rep)} (2OCJ used the same 37 positions)")

def vif2(df):
    X = sm.add_constant(df[["patch_score", "rsa"]].astype(float).values)
    return variance_inflation_factor(X, 1)

def fisher(df):
    a = int((df.destabilizing & df.in_patch).sum()); b = int((df.destabilizing & ~df.in_patch).sum())
    c = int((~df.destabilizing & df.in_patch).sum()); d = int((~df.destabilizing & ~df.in_patch).sum())
    o = odds_ratio([[a, b], [c, d]]); ci = o.confidence_interval(0.95)
    _, p = fisher_exact([[a, b], [c, d]])
    stat = o.statistic if np.isfinite(o.statistic) else None
    return (a, b, c, d), stat, (ci.low, ci.high), p

print("\n=== Collinearity: patch_score vs RSA ===")
for name, d in [("exp n=37 (2XWR)", rep), ("full n=194 (2XWR)", full)]:
    sr, _ = spearmanr(d.patch_score, d.rsa); pr, _ = pearsonr(d.patch_score, d.rsa)
    print(f"  {name}: Spearman={sr:+.3f} Pearson={pr:+.3f} VIF={vif2(d):.2f}")
print("  (2OCJ, for comparison: exp n=37 Spearman=-0.84, VIF=2.75; full n=194 Spearman=-0.77, VIF=2.17)")

tab, orr, ci, pf = fisher(rep)
rho, prho = spearmanr(rep.abs_rep_ddg, rep.patch_score)
y = rep.destabilizing.astype(int).values
l1 = sm.Logit(y, sm.add_constant(rep[["in_patch"]].astype(float))).fit(disp=0)
l2 = sm.Logit(y, sm.add_constant(rep[["in_patch", "rsa"]].astype(float))).fit(disp=0)
o1 = sm.OLS(rep.abs_rep_ddg, sm.add_constant(rep[["patch_score"]])).fit()
o2 = sm.OLS(rep.abs_rep_ddg, sm.add_constant(rep[["patch_score", "rsa"]])).fit()

print(f"\n=== 2XWR burial adjustment (n=37) ===")
print(f"  Q1 Fisher: table={tab} OR={orr:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] p={pf:.4g}")
print(f"  Q1 logistic patch OR: unadj={np.exp(l1.params['in_patch']):.2f} (p={l1.pvalues['in_patch']:.3g}) "
      f"RSA-adj={np.exp(l2.params['in_patch']):.2f} (p={l2.pvalues['in_patch']:.3g})")
print(f"  Q2 Spearman rho={rho:.3f} p={prho:.4g}")
print(f"  Q2 OLS patch coef: unadj={o1.params.iloc[1]:.4f} (p={o1.pvalues.iloc[1]:.3g}) "
      f"RSA-adj={o2.params.iloc[1]:.4f} (p={o2.pvalues.iloc[1]:.3g})")

buried = rep[rep.rsa < RSA_CUT]
tab_b, orr_b, ci_b, pf_b = fisher(buried) if len(buried) >= 4 else (None, None, None, None)
rho_b, prho_b = spearmanr(buried.abs_rep_ddg, buried.patch_score) if len(buried) >= 4 else (np.nan, np.nan)
print(f"\n  Buried-only (RSA<{RSA_CUT}, n={len(buried)}): "
      f"Q1 OR={orr_b if orr_b is None else round(orr_b,2)} (p={pf_b if pf_b is None else round(pf_b,3)})  "
      f"Q2 rho={rho_b:.3f} (p={prho_b:.3g})" if len(buried) >= 4 else f"\n  Buried-only: n={len(buried)}, too few to test")

print(f"\n=== For comparison, 2OCJ (main text, Table tab:adjust) ===")
print(f"  Q1: unadj OR=8.6 (p=0.007) -> RSA-adj OR=3.0 (p=0.27)")
print(f"  Q2: unadj coef=0.033 (p=0.001) -> RSA-adj coef=0.003 (p=0.84)")
print(f"  Buried-only: Q1 OR=3.4 (p=0.53); Q2 rho=0.03 (p=0.91)")

pd.DataFrame([{
    "structure": "2XWR", "n": len(rep),
    "collinearity_spearman_exp": round(spearmanr(rep.patch_score, rep.rsa)[0], 3),
    "collinearity_spearman_full": round(spearmanr(full.patch_score, full.rsa)[0], 3),
    "VIF_exp": round(vif2(rep), 2), "VIF_full": round(vif2(full), 2),
    "Q1_unadj_OR": round(float(np.exp(l1.params["in_patch"])), 2),
    "Q1_unadj_p": round(float(l1.pvalues["in_patch"]), 5),
    "Q1_adj_OR": round(float(np.exp(l2.params["in_patch"])), 2),
    "Q1_adj_p": round(float(l2.pvalues["in_patch"]), 5),
    "Q2_unadj_coef": round(o1.params.iloc[1], 4), "Q2_unadj_p": round(o1.pvalues.iloc[1], 5),
    "Q2_adj_coef": round(o2.params.iloc[1], 4), "Q2_adj_p": round(o2.pvalues.iloc[1], 5),
    "n_buried": len(buried),
    "buried_Q1_OR": None if orr_b is None else round(orr_b, 2),
    "buried_Q1_p": None if pf_b is None else round(pf_b, 4),
    "buried_Q2_rho": round(rho_b, 3) if not np.isnan(rho_b) else None,
    "buried_Q2_p": round(prho_b, 4) if not np.isnan(prho_b) else None,
}]).to_csv(RES / "burial_confound_2xwr.csv", index=False)
print("\nSaved -> results/burial_confound_2xwr.csv")
