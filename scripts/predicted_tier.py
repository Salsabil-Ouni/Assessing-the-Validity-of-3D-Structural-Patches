"""
TIER 3 - FoldX-predicted destabilisation (PositionScan), used to extend coverage
from 37 experimental residues to all 194 structural residues, then repeat Q1/Q2.

Predicted per-residue destabilisation = MEAN FoldX PositionScan ddG over the 19
standard non-self substitutions (FoldX convention: +ve = destabilising). This is a
"mutational sensitivity" score.

IMPORTANT CAVEAT (stated, not hidden): the patch scores are built from the FoldX
alanine scan, and the alanine substitution is one of the 20 in PositionScan, so the
predicted outcome and the patch predictor share a FoldX/structure basis. Tier 3 is
therefore SEMI-INDEPENDENT - it tests coverage + internal consistency, NOT an
independent experimental confirmation. The RSA-adjusted test is the key read-out.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RES = ROOT / "results"
PRED_CUT = 1.0   # predicted-destabilising if mean PositionScan ddG > 1.0 kcal/mol

ala = pd.read_csv(RES / "ala_scan.csv")
wt1 = dict(zip(ala.position, ala.residue))
full = pd.read_csv(RES / "residue_table_full.csv")           # 194: patch_score, rsa, in_patch
exp  = pd.read_csv(RES / "variant_microenvironment.csv")     # 37: rep_ddg

# ---- parse PositionScan output -> mean predicted ddG per residue ----------
out = (RES / "foldx_raw" / "positionscan_scanning_output.txt").read_text().splitlines()
line_re = re.compile(r"^([A-Z0-9]+)A(\d+)([A-Za-z])\s+([-\d.eE]+)$")
bag = {}
for ln in out:
    mobj = line_re.match(ln.strip())
    if not mobj:
        continue
    pos, mut, ddg = int(mobj.group(2)), mobj.group(3), float(mobj.group(4))
    if not mut.isupper():        # skip lowercase special variants (e.g. phospho)
        continue
    if mut == wt1.get(pos):      # skip self (WT->WT)
        continue
    bag.setdefault(pos, []).append(ddg)

pred = pd.DataFrame([{"position": p, "pred_ddg_mean": round(np.mean(v), 3),
                      "pred_ddg_max": round(np.max(v), 3), "n_sub": len(v)}
                     for p, v in bag.items()])
df = full.merge(pred, on="position", how="inner")
df["pred_destab"] = df["pred_ddg_mean"] > PRED_CUT
df.to_csv(RES / "residue_table_full.csv", index=False)
print(f"Parsed PositionScan for {len(pred)} residues; merged table n={len(df)}")

# ---- VALIDATION: FoldX-predicted vs experimental at the 37 ----------------
val = df.merge(exp[["position", "rep_ddg", "abs_rep_ddg"]], on="position", how="inner")
# experimental: negative = destabilising; predicted: positive = destabilising -> expect NEGATIVE corr
r_pe, p_pe = spearmanr(val["pred_ddg_mean"], val["rep_ddg"])
print(f"\nVALIDATION (n={len(val)}): Spearman(pred_ddg_mean, experimental rep_ddg) = {r_pe:.3f} (p={p_pe:.3g})")
print(f"  (expected NEGATIVE: higher predicted destabilisation <-> more negative experimental ddG)")

# ---- TIER 3 Q1 (enrichment) on all 194 ------------------------------------
a = int(( df.pred_destab &  df.in_patch).sum()); b = int(( df.pred_destab & ~df.in_patch).sum())
c = int((~df.pred_destab &  df.in_patch).sum()); d = int((~df.pred_destab & ~df.in_patch).sum())
orr = odds_ratio([[a, b], [c, d]]); ci = orr.confidence_interval(0.95)
_, pf = fisher_exact([[a, b], [c, d]])

# ---- TIER 3 Q2 (intensity) on all 194 -------------------------------------
rho, prho = spearmanr(df["pred_ddg_mean"], df["patch_score"])
# RSA-adjusted OLS: pred_ddg_mean ~ patch_score + rsa
o1 = sm.OLS(df["pred_ddg_mean"], sm.add_constant(df[["patch_score"]])).fit()
o2 = sm.OLS(df["pred_ddg_mean"], sm.add_constant(df[["patch_score", "rsa"]])).fit()
# RSA-adjusted logistic for Q1
y = df.pred_destab.astype(int).values
l1 = sm.Logit(y, sm.add_constant(df[["in_patch"]].astype(float))).fit(disp=0)
l2 = sm.Logit(y, sm.add_constant(df[["in_patch", "rsa"]].astype(float))).fit(disp=0)

print("\n=== TIER 3 (predicted, n=194) ===")
print(f"predicted-destabilising residues (mean ddG > {PRED_CUT}): {int(df.pred_destab.sum())}/{len(df)}")
print(f"Q1 Fisher: [[{a},{b}],[{c},{d}]]  OR={orr.statistic:.2f} CI[{ci.low:.2f},{ci.high:.2f}] p={pf:.3g}")
print(f"Q1 logistic in_patch OR: unadj={np.exp(l1.params[1]):.2f}(p={l1.pvalues[1]:.3g}) | "
      f"RSA-adj={np.exp(l2.params[1]):.2f}(p={l2.pvalues[1]:.3g})")
print(f"Q2 Spearman(pred_ddg_mean, patch_score): rho={rho:.3f} p={prho:.3g}")
print(f"Q2 OLS patch coef: unadj={o1.params[1]:.4f}(p={o1.pvalues[1]:.3g}) | "
      f"RSA-adj={o2.params[1]:.4f}(p={o2.pvalues[1]:.3g})")

# ---- report ---------------------------------------------------------------
rep = f"""# TIER 3 - FoldX-predicted destabilisation (PositionScan), coverage 37 -> 194

**Predicted outcome:** mean FoldX PositionScan ddG over the 19 non-self substitutions
per residue (kcal/mol; +ve = destabilising). Predicted-destabilising = mean > {PRED_CUT}.

**Caveat (semi-circularity):** patch scores derive from the FoldX alanine scan, which
is one of the 20 substitutions in PositionScan, so predictor and outcome share a
FoldX/structure basis. Tier 3 therefore measures coverage + internal consistency, not
independent confirmation. Read the RSA-adjusted results, not the raw association.

## Validation against experiment (n={len(val)})
Spearman(predicted mean ddG, experimental ddG) = **{r_pe:.2f}** (p = {p_pe:.3g}).
Negative as expected (FoldX predicts more destabilisation where experiments measure
more negative ddG), supporting that the FoldX structural predictor is meaningful.

## Q1 - enrichment (n=194)
- predicted-destabilising in/out of patch: [[{a},{b}],[{c},{d}]]
- Fisher OR = **{orr.statistic:.2f}**, 95% CI [{ci.low:.2f}, {ci.high:.2f}], p = {pf:.3g}
- logistic in_patch OR: unadjusted {np.exp(l1.params[1]):.2f} (p={l1.pvalues[1]:.3g});
  **RSA-adjusted {np.exp(l2.params[1]):.2f} (p={l2.pvalues[1]:.3g})**

## Q2 - intensity (n=194)
- Spearman(predicted mean ddG, patch_score) = **{rho:.2f}** (p = {prho:.3g})
- OLS patch coef: unadjusted {o1.params[1]:.4f} (p={o1.pvalues[1]:.3g});
  **RSA-adjusted {o2.params[1]:.4f} (p={o2.pvalues[1]:.3g})**

## Reading
Compare the RSA-adjusted patch effect here (n=194) with Tier 1 (n=37). If it remains
significant after RSA adjustment AND the validation holds, there is support for a
microenvironment effect beyond burial; if it collapses under RSA adjustment as in
Tier 1, the hotspot effect is again attributable largely to burial. Interpret the raw
(unadjusted) association cautiously given the semi-circularity noted above.
"""
(RES / "REPORT_tier3.md").write_text(rep, encoding="utf-8")
print("\nSaved -> results/REPORT_tier3.md and updated residue_table_full.csv")
