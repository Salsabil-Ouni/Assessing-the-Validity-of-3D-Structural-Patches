"""Check G: Q2 sensitivity with the two stabilising residues (N239Y, N268D) excluded from |ddG|."""
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
STAB = [239, 268]

exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
print("full n =", len(exp))
sub = exp[~exp.position.isin(STAB)]
print("excluding the 2 stabilising residues, n =", len(sub))

for name, tab in (("all 37 residues", exp), ("destabilising/neutral only (35 residues)", sub)):
    rho, p = spearmanr(tab.patch_score, tab.abs_rep_ddg)
    print(f"  {name:42s} Q2 Spearman(patch score, |ddG|) = {rho:+.3f} (p={p:.3g})")

pd.DataFrame([
    dict(subset="all 37", n=len(exp), spearman=round(spearmanr(exp.patch_score, exp.abs_rep_ddg)[0], 3),
         p=round(spearmanr(exp.patch_score, exp.abs_rep_ddg)[1], 4)),
    dict(subset="stabilising excluded", n=len(sub), spearman=round(spearmanr(sub.patch_score, sub.abs_rep_ddg)[0], 3),
         p=round(spearmanr(sub.patch_score, sub.abs_rep_ddg)[1], 4)),
]).to_csv(Path(__file__).resolve().parent / "resultsG_q2_sensitivity.csv", index=False)
print("\nsaved resultsG_q2_sensitivity.csv")
