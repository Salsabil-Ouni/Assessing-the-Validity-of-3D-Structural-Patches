"""Check A: density-normalised and RSA-residualised patch score, exactly reproducing the neighbour graph."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr
from Bio.PDB import PDBParser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DIST = cfg["cutoffs"]["patch_distance_angstrom"]
CHAIN = cfg["structure"]["chain"]
PDB = ROOT / cfg["structure"]["repaired_pdb"]

full0 = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position", "ala_ddg"]]
ala_ddg = dict(zip(full0.position, full0.ala_ddg))
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg and r.id[0] == " "}
positions = sorted(heavy)
neigh = {p: {p} for p in positions}
for p, q in itertools.combinations(positions, 2):
    if np.sqrt(((heavy[p][:, None, :] - heavy[q][None, :, :]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)
sum_score = {p: sum(ala_ddg[m] for m in neigh[p]) for p in positions}
mean_score = {p: np.mean([ala_ddg[m] for m in neigh[p]]) for p in positions}
n_neigh = {p: len(neigh[p]) for p in positions}

full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position", "rsa", "patch_score"]]
full["sum_check"] = full.position.map(sum_score)
assert np.allclose(full.patch_score, full.sum_check, atol=1e-3), "graph does not reproduce thesis patch scores"
full["mean_score"] = full.position.map(mean_score)
full["n_neigh"] = full.position.map(n_neigh)
r_sc, _ = spearmanr(full.patch_score, full.n_neigh)
print(f"context: Spearman(sum patch score, neighbourhood size) = {r_sc:+.3f}  (n=194)")

resid_ols = sm.OLS(full.patch_score, sm.add_constant(full.rsa)).fit()
full["resid_score"] = resid_ols.resid
print(f"context: OLS patch_score ~ RSA, R2={resid_ols.rsquared:.3f} (fraction of the raw score explained by RSA alone)")

exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")[["position", "abs_rep_ddg"]]
rows = []
for label, n_set, tab in (("n=37 (experimental)", 37, exp.merge(full, on="position")),
                          ("n=194 (FoldX-extended, own alanine-scan value as outcome)", 194,
                           full.assign(abs_rep_ddg=full.position.map(ala_ddg)))):
    print(f"\n=== {label} ===")
    for name, col in (("sum (current thesis score)", "patch_score"), ("mean (density-normalised)", "mean_score"),
                      ("residual after regressing out RSA", "resid_score")):
        rho, p = spearmanr(tab[col], tab.abs_rep_ddg)
        o1 = sm.OLS(tab.abs_rep_ddg, sm.add_constant(tab[[col]])).fit()
        o2 = sm.OLS(tab.abs_rep_ddg, sm.add_constant(tab[[col, "rsa"]])).fit()
        rows.append(dict(dataset=label, score=name, unadj_spearman=round(rho, 4), unadj_p=round(p, 4),
                         unadj_coef=round(o1.params[col], 4), unadj_coef_p=round(o1.pvalues[col], 4),
                         adj_coef=round(o2.params[col], 4), adj_coef_p=round(o2.pvalues[col], 4),
                         adj_rsa_coef=round(o2.params.rsa, 4), adj_rsa_p=round(o2.pvalues.rsa, 4),
                         corr_with_rsa=round(pearsonr(tab[col], tab.rsa)[0], 3)))
        print(f"  {name:36s} unadj rho={rho:+.3f} (p={p:.3g}) | unadj coef p={o1.pvalues[col]:.3g} "
              f"| ADJUSTED FOR RSA: coef p={o2.pvalues[col]:.3g}, RSA coef p={o2.pvalues.rsa:.3g} "
              f"| corr(score,RSA)={pearsonr(tab[col], tab.rsa)[0]:+.3f}")

pd.DataFrame(rows).to_csv(HERE / "resultsA_density_patch.csv", index=False)
print("\nsaved resultsA_density_patch.csv")
