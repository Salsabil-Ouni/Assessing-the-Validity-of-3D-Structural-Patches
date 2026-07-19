"""
FUNCTIONAL-IMPACT EXTENSION (SEPARATE from the stability result).

Outcome = Giacomelli 2018 combined loss-of-function score (MaveDB 00000068-0-1).
Higher score = more loss of function. This is FUNCTION, not folding stability -
kept entirely separate. It conflates stability with DNA-binding/expression, but it
covers ~175 residues (vs 37), giving real statistical power, especially for the
buried-only 'beyond burial' test.

Structural predictors (patch_score, in_patch, RSA) come from the stability pipeline
(chain A + zinc, 2OCJ). We test whether the STRUCTURAL microenvironment predicts
FUNCTIONAL impact.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]           # project root
FE = ROOT / "functional_ext"
RSA_CUT = 0.10

fun = pd.read_csv(FE / "data" / "giacomelli_missense.csv")     # position, wt, mut, score (high=LOF)
struct = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")  # position, patch_score, rsa, in_patch

# collapse to residue: mean (position sensitivity) + max (most damaging)
res = (fun.groupby("position", as_index=False)
          .agg(func_mean=("score", "mean"), func_max=("score", "max"), n_sub=("score", "size")))
df = res.merge(struct[["position", "patch_score", "rsa", "in_patch"]], on="position", how="inner")
df = df.dropna(subset=["rsa"])
print(f"residues overlapping structure (96-270): {len(df)}  (vs 37 experimental)")

# 'damaging' cutoff for Q1 (report a couple; use mean-score > 0.5 as primary)
CUT = 0.5
df["damaging"] = df["func_mean"] > CUT
print(f"damaging cutoff func_mean>{CUT}: {int(df.damaging.sum())} damaging / {len(df)}")

def fisher(d):
    a=int((d.damaging&d.in_patch).sum()); b=int((d.damaging&~d.in_patch).sum())
    c=int((~d.damaging&d.in_patch).sum()); dd=int((~d.damaging&~d.in_patch).sum())
    if min(a+b,c+dd,a+c,b+dd)==0: return (a,b,c,dd),None,None,None
    o=odds_ratio([[a,b],[c,dd]]); ci=o.confidence_interval(0.95); _,p=fisher_exact([[a,b],[c,dd]])
    return (a,b,c,dd),o.statistic,(ci.low,ci.high),p

buried = df[df.rsa<RSA_CUT]
print(f"\nCONFOUNDING: corr(patch_score,RSA) full n={len(df)}: Spearman={spearmanr(df.patch_score,df.rsa)[0]:+.2f}")

print("\n=== Q1 ENRICHMENT (functional damaging vs in_patch) ===")
for nm,d in [("FULL",df),("BURIED-only",buried)]:
    tab,orr,ci,p=fisher(d)
    if orr: print(f"  {nm} n={len(d)}: table{tab} OR={orr:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] p={p:.3g}")
    else:   print(f"  {nm} n={len(d)}: table{tab} -> undefined (zero cell)")

print("\n=== Q2 INTENSITY (functional impact vs patch_score) ===")
for nm,d in [("FULL",df),("BURIED-only",buried)]:
    r,p=spearmanr(d.func_mean,d.patch_score)
    print(f"  {nm} n={len(d)}: Spearman rho={r:+.3f} p={p:.3g}")

# RSA-adjusted (the key 'beyond burial' models, now with power)
X2=sm.add_constant(df[["patch_score","rsa"]]); o2=sm.OLS(df.func_mean,X2).fit()
o1=sm.OLS(df.func_mean,sm.add_constant(df[["patch_score"]])).fit()
print("\n=== Q2 RSA-adjusted (OLS: func_mean ~ patch_score [+ rsa]) ===")
print(f"  patch coef: unadjusted={o1.params.iloc[1]:.4f} (p={o1.pvalues.iloc[1]:.3g}) | "
      f"RSA-adjusted={o2.params.iloc[1]:.4f} (p={o2.pvalues.iloc[1]:.3g})")

df.to_csv(FE / "results" / "functional_residue_table.csv", index=False)

rep=f"""# FUNCTIONAL-IMPACT EXTENSION (separate from the stability result)

**Outcome:** Giacomelli 2018 combined loss-of-function score (MaveDB urn:00000068-0-1);
higher = more loss of function. **This is FUNCTION, not folding stability** - it mixes
stability with DNA-binding/expression, so it is reported separately and NOT merged with
the 37-residue ΔΔG result. Its value is coverage/power: **{len(df)} residues** (96-270).

**Structural predictors** (patch_score, in_patch, RSA) are from the stability pipeline.

## Q1 - enrichment (functionally-damaging vs in hotspot patch)
Full (n={len(df)}) and buried-only (n={len(buried)}) - see printed table; OR + 95% CI + Fisher p.

## Q2 - intensity (functional impact vs microenvironment score)
- Full: Spearman rho, p (printed)
- Buried-only (the powered 'beyond burial' test): Spearman rho, p (printed)
- RSA-adjusted OLS: patch coefficient before/after adding RSA (printed)

## How to read it
If the patch effect on FUNCTIONAL impact survives RSA adjustment / holds within buried
residues (now with ~{len(buried)} buried residues, not 18), that is evidence the structural
microenvironment predicts functional impact beyond burial. Interpret cautiously: functional
score is a confounded proxy, and this answers a DIFFERENT question than the stability hypothesis.
"""
(FE / "results" / "REPORT_functional.md").write_text(rep, encoding="utf-8")
print(f"\nSaved -> functional_ext/results/ (table + REPORT_functional.md)")
