"""Check F: burial confounder re-derived with RSA computed on the full four-chain 2OCJ assembly."""
from pathlib import Path
import numpy as np
import pandas as pd
import freesasa
import statsmodels.api as sm
from scipy.stats import spearmanr, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
CHAIN = "A"
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,
          "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}

ala = pd.read_csv(ROOT / "results" / "ala_scan.csv")
res_wt = dict(zip(ala.position, ala.residue))

s = freesasa.Structure(str(ROOT / "data" / "candidates" / "2OCJ.pdb"))
result = freesasa.calc(s)
areas = result.residueAreas()
print("chains in the raw deposited assembly:", sorted(areas.keys()))
chainA_assembly = {int(k): v.total for k, v in areas[CHAIN].items() if k.isdigit()}
assembly_rsa = {p: chainA_assembly[p] / MAXASA.get(res_wt.get(p, ""), 999)
                for p in chainA_assembly if p in res_wt}
print(f"assembly RSA computed for {len(assembly_rsa)} chain-A residues (raw, unrepaired coordinates)")

full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position", "rsa", "patch_score"]]
full["assembly_rsa"] = full.position.map(assembly_rsa)
missing = full.assembly_rsa.isna().sum()
print(f"positions with no match in the raw assembly (kept at monomer RSA... excluded here instead): {missing}")
full = full.dropna(subset=["assembly_rsa"])
full["delta_rsa"] = full.rsa - full.assembly_rsa  # positive = looked MORE exposed in isolated chain than in assembly
print(f"\ncorr(monomer RSA, assembly RSA) across {len(full)} residues: rho={spearmanr(full.rsa, full.assembly_rsa)[0]:+.3f}")
interface_like = full[full.delta_rsa > 0.05]
print(f"residues where isolated-chain RSA exceeds assembly RSA by >0.05 (candidate interface positions): "
      f"{len(interface_like)} of {len(full)}")

exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
exp = exp.merge(full[["position", "assembly_rsa"]], on="position", how="inner")
print(f"\nof the 37 experimental residues, {(exp.position.isin(interface_like.position)).sum()} are candidate "
      f"interface positions by the >0.05 rule; assembly RSA available for {len(exp)} of 37")

rows = []
for rsa_col, rsa_label in (("rsa", "monomer RSA (thesis)"), ("assembly_rsa", "assembly RSA (this check)")):
    y = exp.abs_rep_ddg
    o_unadj = sm.OLS(y, sm.add_constant(exp[["patch_score"]])).fit()
    o_adj = sm.OLS(y, sm.add_constant(exp[["patch_score", rsa_col]])).fit()
    m1 = sm.Logit(exp.destabilizing.astype(float), sm.add_constant(exp[["in_patch"]].astype(float))).fit(disp=0)
    m2 = sm.Logit(exp.destabilizing.astype(float), sm.add_constant(exp[["in_patch", rsa_col]].astype(float))).fit(disp=0)
    rows.append(dict(rsa_used=rsa_label, n=len(exp),
                     Q2_unadj_patch_p=round(o_unadj.pvalues.patch_score, 4),
                     Q2_adj_patch_coef=round(o_adj.params.patch_score, 4), Q2_adj_patch_p=round(o_adj.pvalues.patch_score, 4),
                     Q2_adj_rsa_coef=round(o_adj.params[rsa_col], 4), Q2_adj_rsa_p=round(o_adj.pvalues[rsa_col], 4),
                     Q1_unadj_OR=round(np.exp(m1.params.in_patch), 3), Q1_unadj_p=round(m1.pvalues.in_patch, 4),
                     Q1_adj_OR=round(np.exp(m2.params.in_patch), 3), Q1_adj_p=round(m2.pvalues.in_patch, 4)))
    print(f"\n=== RSA source: {rsa_label} (n={len(exp)}) ===")
    print(f"  Q2 unadjusted patch coefficient p = {o_unadj.pvalues.patch_score:.4g}")
    print(f"  Q2 adjusted: patch coef {o_adj.params.patch_score:+.4f} (p={o_adj.pvalues.patch_score:.4g}), "
          f"{rsa_col} coef {o_adj.params[rsa_col]:+.4f} (p={o_adj.pvalues[rsa_col]:.4g})")
    print(f"  Q1 unadjusted OR = {np.exp(m1.params.in_patch):.2f} (p={m1.pvalues.in_patch:.4g})")
    print(f"  Q1 adjusted OR   = {np.exp(m2.params.in_patch):.2f} (p={m2.pvalues.in_patch:.4g})")

pd.DataFrame(rows).to_csv(HERE / "resultsF_assembly_rsa.csv", index=False)
print("\nsaved resultsF_assembly_rsa.csv")
