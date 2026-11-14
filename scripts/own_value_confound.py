"""
A second confound check, parallel to the RSA/burial analysis (extended_analysis.py,
consistent_context.py): does the spatially-aggregated PATCH score add anything beyond
the residue's OWN alanine-scan value (no neighbour contribution at all)?

Motivated by the distance-sweep negative control (distance_sweep.py, D=1.0 A): with no
neighbours in range, the "microenvironment score" collapses to the residue's own ala_ddg,
and Q2 there (rho=0.55) is at least as strong as the full 8-A patch score (rho=0.51). This
tests that observation properly, on the same n=37 experimental set, using only quantities
already computed by the main pipeline (ala_scan_cA.csv, residue_table_cA.csv) - no new
structure or metric, unlike the appendix.

Q1: is patch membership (in_patch, 8 A) enriched for destabilisation beyond simply being a
    hotspot oneself (own ala_ddg >= threshold, no spatial step at all)?
Q2: does the patch score predict |ddG| beyond the residue's own ala_ddg, in a joint OLS?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
HOT = cfg["cutoffs"]["ala_hotspot_ddg"]

ala = pd.read_csv(RES / "ala_scan_cA.csv")[["position", "ddG"]].rename(columns={"ddG": "ala_ddg"})
exp = pd.read_csv(RES / "residue_table_cA.csv")  # rep_ddg, abs_rep_ddg, destabilizing, patch_score, rsa, in_patch
d = exp.merge(ala, on="position", how="inner")
d["own_hotspot"] = d.ala_ddg >= HOT
assert len(d) == len(exp), "merge dropped rows"

print(f"n = {len(d)}")

# ---- Q2: OLS nesting -------------------------------------------------------
r_patch, p_patch = spearmanr(d.abs_rep_ddg, d.patch_score)
r_own, p_own = spearmanr(d.abs_rep_ddg, d.ala_ddg)

o_patch = sm.OLS(d.abs_rep_ddg, sm.add_constant(d[["patch_score"]])).fit()
o_own = sm.OLS(d.abs_rep_ddg, sm.add_constant(d[["ala_ddg"]])).fit()
o_joint = sm.OLS(d.abs_rep_ddg, sm.add_constant(d[["patch_score", "ala_ddg"]])).fit()

print("\n=== Q2: does the 8-A patch score add anything beyond the residue's own value? ===")
print(f"  Spearman(|ddG|, patch_score) alone: rho={r_patch:.3f} p={p_patch:.4g}")
print(f"  Spearman(|ddG|, own ala_ddg)  alone: rho={r_own:.3f} p={p_own:.4g}")
print(f"  OLS |ddG| ~ patch_score            : coef={o_patch.params.iloc[1]:.4f} p={o_patch.pvalues.iloc[1]:.4g} R2={o_patch.rsquared:.3f}")
print(f"  OLS |ddG| ~ own ala_ddg            : coef={o_own.params.iloc[1]:.4f} p={o_own.pvalues.iloc[1]:.4g} R2={o_own.rsquared:.3f}")
print(f"  OLS |ddG| ~ patch_score + ala_ddg  : patch coef={o_joint.params['patch_score']:.4f} p={o_joint.pvalues['patch_score']:.4g}")
print(f"                                       own   coef={o_joint.params['ala_ddg']:.4f} p={o_joint.pvalues['ala_ddg']:.4g}  R2={o_joint.rsquared:.3f}")
print(f"  (R2 alone={max(o_patch.rsquared, o_own.rsquared):.3f} -> joint={o_joint.rsquared:.3f}: "
      f"delta R2 from adding the weaker term = {o_joint.rsquared - max(o_patch.rsquared, o_own.rsquared):.4f})")

# ---- Q1: Fisher nesting -----------------------------------------------------
def fisher(dest, flag):
    a = int((dest & flag).sum()); b = int((dest & ~flag).sum())
    c = int((~dest & flag).sum()); d_ = int((~dest & ~flag).sum())
    o = odds_ratio([[a, b], [c, d_]]); ci = o.confidence_interval(0.95)
    _, p = fisher_exact([[a, b], [c, d_]])
    return (a, b, c, d_), o.statistic, (ci.low, ci.high), p

print("\n=== Q1: is 8-A patch membership enriched beyond simply being a hotspot oneself? ===")
tab_p, or_p, ci_p, p_p = fisher(d.destabilizing, d.in_patch)
print(f"  destabilising ~ in_patch (8 A patch)   : table={tab_p} OR={or_p:.2f} CI[{ci_p[0]:.2f},{ci_p[1]:.2f}] p={p_p:.4g}")
tab_h, or_h, ci_h, p_h = fisher(d.destabilizing, d.own_hotspot)
print(f"  destabilising ~ own_hotspot (no neighbours): table={tab_h} OR={or_h:.2f} CI[{ci_h[0]:.2f},{ci_h[1]:.2f}] p={p_h:.4g}")

if (d.in_patch == d.own_hotspot).all():
    print("  NOTE: in_patch and own_hotspot are IDENTICAL on this n=37 sample (tables match exactly).")
    print("  This is structural, not a new finding: a patch (size>=2 component) is by construction a set")
    print("  of hotspots, so in_patch implies own_hotspot always; the converse fails only for the single")
    print("  globally isolated hotspot (residue 226, a size-1 component), which is not among these 37")
    print("  positions. A joint logistic model is therefore singular (perfectly collinear regressors) and")
    print("  is not run: Q1, as a BINARY test, cannot distinguish 'being a hotspot' from 'being in a patch'")
    print("  on this sample - only the continuous Q2 score (which differs residue-by-residue) can test")
    print("  whether spatial aggregation adds anything, which is why the OLS analysis above is the")
    print("  decisive test here, not this Q1 side.")
    l_joint_patch_OR, l_joint_patch_p = or_p, p_p
    l_joint_own_OR, l_joint_own_p = or_p, p_p
else:
    y = d.destabilizing.astype(int).values
    l_joint = sm.Logit(y, sm.add_constant(d[["in_patch", "own_hotspot"]].astype(float))).fit(disp=0)
    l_joint_patch_OR = float(np.exp(l_joint.params["in_patch"])); l_joint_patch_p = float(l_joint.pvalues["in_patch"])
    l_joint_own_OR = float(np.exp(l_joint.params["own_hotspot"])); l_joint_own_p = float(l_joint.pvalues["own_hotspot"])
    print(f"  logistic OR, joint model: in_patch = {l_joint_patch_OR:.2f} (p={l_joint_patch_p:.3g}), "
          f"own_hotspot = {l_joint_own_OR:.2f} (p={l_joint_own_p:.3g})")

pd.DataFrame([{
    "Q2_rho_patch": round(r_patch, 3), "Q2_p_patch": round(p_patch, 5),
    "Q2_rho_own": round(r_own, 3), "Q2_p_own": round(p_own, 5),
    "Q2_R2_patch": round(o_patch.rsquared, 3), "Q2_R2_own": round(o_own.rsquared, 3),
    "Q2_R2_joint": round(o_joint.rsquared, 3),
    "Q2_joint_patch_coef": round(o_joint.params["patch_score"], 4),
    "Q2_joint_patch_p": round(o_joint.pvalues["patch_score"], 5),
    "Q2_joint_own_coef": round(o_joint.params["ala_ddg"], 4),
    "Q2_joint_own_p": round(o_joint.pvalues["ala_ddg"], 5),
    "Q1_OR_patch": round(or_p, 2), "Q1_p_patch": round(p_p, 5),
    "Q1_OR_own": round(or_h, 2), "Q1_p_own": round(p_h, 5),
    "Q1_identical_on_sample": bool((d.in_patch == d.own_hotspot).all()),
    "Q1_joint_patch_OR": round(l_joint_patch_OR, 2), "Q1_joint_patch_p": round(l_joint_patch_p, 5),
    "Q1_joint_own_OR": round(l_joint_own_OR, 2), "Q1_joint_own_p": round(l_joint_own_p, 5),
}]).to_csv(RES / "own_value_confound.csv", index=False)
print("\nSaved -> results/own_value_confound.csv")
