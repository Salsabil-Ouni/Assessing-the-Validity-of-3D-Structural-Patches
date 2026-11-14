"""
"Beyond burial" test using a property burial does NOT summarise: proximity to
the DNA-binding interface (dna_proximity.py, from the DNA-bound 1TUP
structure), rather than re-testing the same FoldX/2OCJ patch score that Section
r-burial already showed is redundant with RSA. This directly follows Chapter 5
Discussion's proposed direction ("redirect the patch framework toward
physicochemical properties burial cannot see, e.g. electrostatic
complementarity at DNA... interfaces").

Simplification (stated explicitly, not hidden): DNA proximity (closest heavy-atom
distance) is used as a cheap structural proxy for interface involvement, not a
full electrostatic-complementarity calculation (which would need PDB2PQR/APBS);
it captures whether a residue is positioned to contact DNA at all, not the
quality of that contact.

Tested on the two functional replication datasets (Giacomelli 2018, Kato 2003),
NOT on the thermodynamic ddG set: ThermoMutDB measures folding stability only,
so DNA proximity has no reason to predict it and is not expected to. Functional
assays conflate stability and DNA-binding, so this is where a DNA-proximity
effect, if real, should appear.

Model: damage ~ patch_score + rsa + dna_proximity_score
  dna_proximity_score = -min_dist_dna (so higher = closer to DNA = expected
  more damaging if a variant there disrupts binding)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "functional_ext"
RES = ROOT / "results"

full = pd.read_csv(RES / "residue_table_full_cA.csv")[["position", "patch_score", "rsa"]]
dna = pd.read_csv(RES / "dna_proximity.csv")[["position", "min_dist_dna"]]
dna["dna_proximity_score"] = -dna.min_dist_dna
struct = full.merge(dna, on="position")

three1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T',
'Trp':'W','Tyr':'Y','Val':'V'}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')
HOTSPOTS = [('R',175,'H'),('R',248,'W'),('R',248,'Q'),('R',249,'S'),
            ('G',245,'S'),('R',273,'H'),('R',282,'W')]

def load(fname):
    df = pd.read_csv(FE / "data" / fname)
    rows = []
    for _, r in df.iterrows():
        m = RX.match(str(r['hgvs_pro']))
        if not m:
            continue
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt in three1 and mut in three1 and wt != mut and pd.notna(r['score']):
            rows.append((pos, three1[wt], three1[mut], float(r['score'])))
    return pd.DataFrame(rows, columns=['position', 'wt', 'mut', 'score'])

def orient(v):
    hs = pd.concat([v[(v.position == p) & (v.wt == w) & (v.mut == m)] for w, p, m in HOTSPOTS])
    if len(hs) == 0:
        return v
    if hs.score.mean() < v.score.mean():
        v = v.copy(); v["score"] = -v["score"]
    return v

rows = []
for name, fname in [("Giacomelli 2018", "giacomelli_scores.csv"), ("Kato 2003", "kato_scores.csv")]:
    v = orient(load(fname))
    d = (v.groupby("position", as_index=False).agg(damage=("score", "mean"))
           .merge(struct, on="position"))

    r_dna, p_dna = spearmanr(d.damage, d.dna_proximity_score)
    r_patch, p_patch = spearmanr(d.damage, d.patch_score)

    m_patch_only = sm.OLS(d.damage, sm.add_constant(d[["patch_score"]])).fit()
    m_patch_rsa = sm.OLS(d.damage, sm.add_constant(d[["patch_score", "rsa"]])).fit()
    m_full = sm.OLS(d.damage, sm.add_constant(d[["patch_score", "rsa", "dna_proximity_score"]])).fit()
    m_dna_only = sm.OLS(d.damage, sm.add_constant(d[["dna_proximity_score"]])).fit()

    print(f"\n=== {name} (n={len(d)}) ===")
    print(f"  Spearman(damage, dna_proximity_score)   rho={r_dna:+.3f}  p={p_dna:.3g}")
    print(f"  Spearman(damage, patch_score)           rho={r_patch:+.3f}  p={p_patch:.3g}")
    print(f"  OLS damage ~ dna_proximity_score alone:  coef={m_dna_only.params.iloc[1]:+.4f} p={m_dna_only.pvalues.iloc[1]:.3g}  R2={m_dna_only.rsquared:.3f}")
    print(f"  OLS damage ~ patch_score alone:          coef={m_patch_only.params.iloc[1]:+.4f} p={m_patch_only.pvalues.iloc[1]:.3g}  R2={m_patch_only.rsquared:.3f}")
    print(f"  OLS damage ~ patch_score + rsa:          patch coef={m_patch_rsa.params.iloc[1]:+.4f} p={m_patch_rsa.pvalues.iloc[1]:.3g}  R2={m_patch_rsa.rsquared:.3f}")
    print(f"  OLS damage ~ patch_score + rsa + dna:    patch coef={m_full.params['patch_score']:+.4f} p={m_full.pvalues['patch_score']:.3g}")
    print(f"                                            rsa   coef={m_full.params['rsa']:+.4f} p={m_full.pvalues['rsa']:.3g}")
    print(f"                                            dna   coef={m_full.params['dna_proximity_score']:+.4f} p={m_full.pvalues['dna_proximity_score']:.3g}  R2={m_full.rsquared:.3f}")

    rows.append({
        "assay": name, "n": len(d),
        "dna_rho": round(r_dna, 3), "dna_rho_p": round(p_dna, 5),
        "patch_rho": round(r_patch, 3), "patch_rho_p": round(p_patch, 5),
        "full_model_patch_coef": round(m_full.params["patch_score"], 4),
        "full_model_patch_p": round(m_full.pvalues["patch_score"], 5),
        "full_model_rsa_coef": round(m_full.params["rsa"], 4),
        "full_model_rsa_p": round(m_full.pvalues["rsa"], 5),
        "full_model_dna_coef": round(m_full.params["dna_proximity_score"], 4),
        "full_model_dna_p": round(m_full.pvalues["dna_proximity_score"], 5),
        "full_model_R2": round(m_full.rsquared, 3),
        "patch_rsa_only_R2": round(m_patch_rsa.rsquared, 3),
    })

out = pd.DataFrame(rows)
out.to_csv(RES / "interface_beyond_burial.csv", index=False)
print("\nSaved -> results/interface_beyond_burial.csv")
