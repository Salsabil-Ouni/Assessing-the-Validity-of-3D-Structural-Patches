"""
Stratified test: restrict to BURIED residues (RSA < 0.1) and ask whether the
hotspot/patch effect survives WITHIN buried residues - i.e. beyond burial itself.

Q1: among buried residues, are destabilising variants enriched in hotspot-patch
    residues vs non-patch residues?  (2x2 Fisher)
Q2: among buried residues, does aggregated patch score correlate with variant |ddG|?  (Spearman)

Run on:
  - experimental residues (n<=37) - PRIMARY (uses measured ddG)
  - FoldX-predicted residues (n<=194) - supplementary, more power (semi-circular; flagged)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RSA_CUT = 0.10

exp  = pd.read_csv(RES / "residue_table_cA.csv")          # measured: rep_ddg, abs_rep_ddg, patch_score, rsa, in_patch, destabilizing
full = pd.read_csv(RES / "residue_table_full_cA.csv")     # predicted: pred_ddg_mean, pred_destab, patch_score, rsa, in_patch

def q1(df, dest_col):
    a=int((df[dest_col]&df.in_patch).sum()); b=int((df[dest_col]&~df.in_patch).sum())
    c=int((~df[dest_col]&df.in_patch).sum()); d=int((~df[dest_col]&~df.in_patch).sum())
    tab=(a,b,c,d)
    if min(a+b,c+d,a+c,b+d)==0:
        return tab, None, None, None, "no contrast (a marginal is zero) -> test undefined"
    o=odds_ratio([[a,b],[c,d]]); ci=o.confidence_interval(0.95); _,p=fisher_exact([[a,b],[c,d]])
    return tab, o.statistic, (ci.low,ci.high), p, ""

def show(name, df, dest_col, mag_col):
    n=len(df); nin=int(df.in_patch.sum())
    print(f"\n=== {name}: buried residues (RSA<{RSA_CUT}) n={n}  (in_patch={nin}, non-patch={n-nin}) ===")
    print(pd.crosstab(df[dest_col], df.in_patch, rownames=[dest_col], colnames=["in_patch"]).to_string())
    tab,orr,ci,p,note = q1(df, dest_col)
    if note:
        print(f"  Q1: table(a,b,c,d)={tab}  -> {note}")
    else:
        print(f"  Q1 Fisher: table(a,b,c,d)={tab}  OR={orr:.2f}  95%CI[{ci[0]:.2f},{ci[1]:.2f}]  p={p:.4g}")
    if df.in_patch.nunique()>1 or True:
        rho,pr = spearmanr(df[mag_col], df.patch_score)
        print(f"  Q2 Spearman(|ddG|, patch_score) within buried: rho={rho:.3f}  p={pr:.4g}  (n={n})")

# experimental (primary)
be = exp[exp.rsa < RSA_CUT].copy()
show("EXPERIMENTAL", be, "destabilizing", "abs_rep_ddg")

# predicted (supplementary, more power)
full["pred_destab"] = full.pred_ddg_mean > 1.0
bf = full[full.rsa < RSA_CUT].copy()
full["abs_pred"] = full.pred_ddg_mean.abs()
bf["abs_pred"] = bf.pred_ddg_mean
show("PREDICTED (FoldX, semi-circular)", bf, "pred_destab", "abs_pred")

# also report counts at a few RSA cutoffs so we know how power changes
print("\n=== buried-residue counts at different RSA cut-offs (experimental set) ===")
for c in [0.05,0.10,0.15,0.20]:
    s=exp[exp.rsa<c]
    print(f"  RSA<{c}: n={len(s)}  in_patch={int(s.in_patch.sum())}  non-patch={len(s)-int(s.in_patch.sum())}  destabilising={int(s.destabilizing.sum())}")
