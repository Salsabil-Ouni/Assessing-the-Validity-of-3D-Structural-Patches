"""
Sensitivity of the conclusion to the destabilising ΔΔG cut-off.
Re-run Q1 (enrichment) at |ΔΔG| cut-offs 0.5/1.0/1.5/2.0 kcal/mol:
  (a) full experimental set, and
  (b) buried residues only (RSA < 0.1) - the 'beyond burial' test.
Q2 (Spearman of |ΔΔG| vs patch score) uses the continuous ΔΔG, so it does NOT
depend on the cut-off - reported once for reference.
"""
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio

ROOT = Path(__file__).resolve().parents[1]
exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")  # rep_ddg, abs_rep_ddg, patch_score, rsa, in_patch
RSA_CUT = 0.10

def q1(df, cut):
    dest = df["rep_ddg"] < -cut          # ThermoMutDB: negative = destabilising
    a = int(( dest &  df.in_patch).sum()); b = int(( dest & ~df.in_patch).sum())
    c = int((~dest &  df.in_patch).sum()); d = int((~dest & ~df.in_patch).sum())
    n_dest = int(dest.sum())
    if min(a+b, c+d, a+c, b+d) == 0:
        return n_dest, (a,b,c,d), None, None, None
    o = odds_ratio([[a,b],[c,d]]); ci = o.confidence_interval(0.95); _, p = fisher_exact([[a,b],[c,d]])
    return n_dest, (a,b,c,d), o.statistic, (ci.low, ci.high), p

buried = exp[exp.rsa < RSA_CUT]
print(f"FULL set n={len(exp)} | BURIED (RSA<{RSA_CUT}) n={len(buried)}\n")
print(f"{'cutoff':>6} | {'n_dest':>6} | {'FULL: OR (95% CI) p':<34} | {'BURIED-ONLY: OR (95% CI) p':<34}")
print("-"*92)
for cut in [0.5, 1.0, 1.5, 2.0]:
    nf, tf, orf, cif, pf = q1(exp, cut)
    nb, tb, orb, cib, pb = q1(buried, cut)
    full_s = f"OR={orf:.1f} [{cif[0]:.1f},{cif[1]:.0f}] p={pf:.3g}" if orf else f"n/a (cell=0) {tf}"
    bur_s  = f"OR={orb:.1f} [{cib[0]:.1f},{cib[1]:.0f}] p={pb:.3g}" if orb else f"n/a (cell=0) {tb}"
    print(f"{cut:>6.1f} | {nf:>6} | {full_s:<34} | {bur_s:<34}")

print("\nQ2 (continuous, cut-off-independent):")
r_all, p_all = spearmanr(exp.abs_rep_ddg, exp.patch_score)
r_bur, p_bur = spearmanr(buried.abs_rep_ddg, buried.patch_score)
print(f"  full set:    Spearman(|ddG|, patch_score) rho={r_all:.3f} p={p_all:.3g}")
print(f"  buried only: Spearman(|ddG|, patch_score) rho={r_bur:.3f} p={p_bur:.3g}")
