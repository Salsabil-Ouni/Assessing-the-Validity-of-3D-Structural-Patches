"""
EXTENDED / SENSITIVITY analysis.

Primary result = the 37 experimentally-measured residues (Tier 1).
Here we test ROBUSTNESS, focused on the central question:

  Does increasing structural coverage reduce the confounding between the
  hotspot-patch score and solvent burial (RSA)?

Confounding metrics (correlation, VIF) do NOT depend on any destabilisation
outcome, so they can be computed cleanly on the full 194-residue structural set
and compared with the 37-residue experimental set. If they are similar, the
collinearity is intrinsic and more data will not de-confound the design.

We also restate Tier-1 Q1/Q2 with and without RSA adjustment.

Tier-3 (predicted outcome) Q1/Q2 are NOT computed from the alanine scan, because
that is circular (patches are built from it); a valid predicted tier needs
independent FoldX mutation predictions (separate job).
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy.stats import spearmanr, pearsonr, fisher_exact, linregress
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import freesasa
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CHAIN = cfg["structure"]["chain"]
HOT   = cfg["cutoffs"]["ala_hotspot_ddg"]
DIST  = cfg["cutoffs"]["patch_distance_angstrom"]
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]
BURIED_REL= cfg["sasa"]["buried_cutoff_relative"]
PDB   = ROOT / cfg["structure"]["repaired_pdb"]
RES   = ROOT / "results"

ala = pd.read_csv(RES / "ala_scan.csv")
ala_ddg = dict(zip(ala.position, ala.ddG))
is_hot  = {p: d >= HOT for p, d in ala_ddg.items()}

# ---- geometry: neighbours within DIST, for ALL 194 residues ---------------
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg}
positions = sorted(heavy)
neigh = {p: {p} for p in positions}
for p, q in itertools.combinations(positions, 2):
    a, b = heavy[p], heavy[q]
    if np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)
agg_score = {p: sum(ala_ddg[m] for m in neigh[p]) for p in positions}

# hotspot patch clusters -> in_patch membership
hot_pos = [p for p in positions if is_hot[p]]
G = nx.Graph(); G.add_nodes_from(hot_pos)
for p, q in itertools.combinations(hot_pos, 2):
    if q in neigh[p]:
        G.add_edge(p, q)
patch_members = set().union(*[c for c in nx.connected_components(G) if len(c) >= 2])

# ---- RSA for ALL residues (freesasa, chain A monomer) ---------------------
class ChainAProt(Select):
    def accept_chain(self, ch): return ch.id == CHAIN
    def accept_residue(self, r): return r.id[0] == " "
io = PDBIO(); io.set_structure(PDBParser(QUIET=True).get_structure("p", str(PDB)))
tmp = RES / "_chainA_all.pdb"; io.save(str(tmp), ChainAProt())
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,
          "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
res_wt = dict(zip(ala.position, ala.residue))
areas = freesasa.calc(freesasa.Structure(str(tmp))).residueAreas()[CHAIN]
rsa = {int(k): v.total / MAXASA.get(res_wt.get(int(k), ""), 999)
       for k, v in areas.items() if k.isdigit()}
tmp.unlink(missing_ok=True)

# ---- full 194-residue structural table ------------------------------------
full = pd.DataFrame({"position": positions})
full["ala_ddg"]     = full.position.map(ala_ddg)
full["patch_score"] = full.position.map(agg_score)
full["rsa"]         = full.position.map(rsa)
full["is_hotspot"]  = full.position.map(is_hot)
full["in_patch"]    = full.position.isin(patch_members)
full = full.dropna(subset=["rsa"])
full.to_csv(RES / "residue_table_full.csv", index=False)

# ---- experimental 37-residue table ----------------------------------------
exp = pd.read_csv(RES / "variant_microenvironment.csv")   # has rep_ddg, rel_sasa, etc.
exp = exp.rename(columns={"rel_sasa": "rsa", "aggregated_patch_score": "patch_score"})

# ===================== CONFOUNDING METRICS =================================
def vif_two(df):
    X = sm.add_constant(df[["patch_score", "rsa"]].astype(float).values)
    return variance_inflation_factor(X, 1), variance_inflation_factor(X, 2)

def confound(df, name):
    sr, sp = spearmanr(df.patch_score, df.rsa)
    pr, pp = pearsonr(df.patch_score, df.rsa)
    vif_ps, vif_rsa = vif_two(df)
    print(f"  [{name}] n={len(df)}")
    print(f"     corr(patch_score, RSA): Spearman={sr:+.3f} (p={sp:.3g}), Pearson={pr:+.3f} (p={pp:.3g})")
    print(f"     VIF: patch_score={vif_ps:.2f}, RSA={vif_rsa:.2f}")
    return dict(name=name, n=len(df), spearman=sr, pearson=pr, vif=vif_ps)

print("=== CONFOUNDING between patch_score and RSA (outcome-independent) ===")
c37  = confound(exp,  "experimental n=37")
c194 = confound(full, "full structural n=194")

# ===================== TIER 1 Q1/Q2 (experimental) =========================
print("\n=== TIER 1 (experimental, n=37) ===")
a = int(( exp.destabilizing &  exp.in_hotspot_patch).sum())
b = int(( exp.destabilizing & ~exp.in_hotspot_patch).sum())
c = int((~exp.destabilizing &  exp.in_hotspot_patch).sum())
d = int((~exp.destabilizing & ~exp.in_hotspot_patch).sum())
orr = odds_ratio([[a, b], [c, d]]); ci = orr.confidence_interval(0.95)
_, pf = fisher_exact([[a, b], [c, d]])
rho, prho = spearmanr(exp.abs_rep_ddg, exp.patch_score)
print(f"  Q1 Fisher: table=[[{a},{b}],[{c},{d}]]  OR={orr.statistic:.2f} CI[{ci.low:.2f},{ci.high:.2f}] p={pf:.4g}")
print(f"  Q2 Spearman(|rep ddG|, patch_score): rho={rho:.3f} p={prho:.4g}")
# RSA-adjusted
y = exp.destabilizing.astype(int).values
Xu = sm.add_constant(exp[["in_patch" if "in_patch" in exp else "in_hotspot_patch"]].astype(float).values)
m1 = sm.Logit(y, sm.add_constant(exp[["in_hotspot_patch"]].astype(float).values)).fit(disp=0)
m2 = sm.Logit(y, sm.add_constant(exp[["in_hotspot_patch", "rsa"]].astype(float).values)).fit(disp=0)
print(f"  Q1 logistic patch OR: unadjusted={np.exp(m1.params[1]):.2f} (p={m1.pvalues[1]:.3g}) | "
      f"RSA-adjusted={np.exp(m2.params[1]):.2f} (p={m2.pvalues[1]:.3g})")
o1 = sm.OLS(exp.abs_rep_ddg.values, sm.add_constant(exp[["patch_score"]].values)).fit()
o2 = sm.OLS(exp.abs_rep_ddg.values, sm.add_constant(exp[["patch_score", "rsa"]].values)).fit()
print(f"  Q2 OLS patch_score coef: unadjusted={o1.params[1]:.4f} (p={o1.pvalues[1]:.3g}) | "
      f"RSA-adjusted={o2.params[1]:.4f} (p={o2.pvalues[1]:.3g})")

# ---- report ---------------------------------------------------------------
def verdict():
    intrinsic = abs(c194["pearson"] - c37["pearson"]) < 0.25
    adj_ns = (m2.pvalues[1] >= 0.05) and (o2.pvalues[1] >= 0.05)
    s = []
    s.append(f"- corr(patch_score, RSA) is {c37['pearson']:.2f} at n=37 and "
             f"{c194['pearson']:.2f} at n=194 -> the collinearity is "
             f"{'INTRINSIC (not a small-sample artifact); increasing coverage does NOT de-confound' if intrinsic else 'somewhat sample-dependent'}.")
    s.append(f"- VIF(patch_score|RSA) = {c37['vif']:.1f} (n=37), {c194['vif']:.1f} (n=194).")
    if adj_ns:
        s.append("- After adjusting for RSA, the patch effect is NOT significant in either Q1 or Q2.")
        s.append("- CONCLUSION: the observed hotspot-patch effect is **largely explained by residue "
                 "burial (RSA)**. We cannot demonstrate a microenvironment effect beyond burial in this "
                 "dataset; this is stated as a limitation.")
    else:
        s.append("- The patch effect persists after RSA adjustment -> evidence for a microenvironment "
                 "effect beyond burial.")
    return "\n".join(s)

report = f"""# TP53 - Extended / sensitivity analysis (robustness of the microenvironment effect)

The **primary result remains the 37 experimentally-measured residues (Tier 1)**.
This document reports robustness analyses.

## Data tiers - what is and is not feasible
- **Tier 1 - experimental (n=37):** ThermoMutDB single-point ddG, region 93-290. Primary.
- **Tier 2 - additional experimental databases:** ThermoMutDB already aggregates ProTherm
  and the thermodynamic literature for p53, so little independent ddG remains. MaveDB
  (functional/abundance scores) and the IARC/TP53 database (mutation prevalence,
  transactivation class) measure DIFFERENT phenotypes, not folding ddG, and cannot be
  merged into a ddG column without conflating quantities. Not performed (would be invalid).
- **Tier 3 - FoldX-predicted ddG:** Using the alanine-scan profile as the predicted outcome
  is CIRCULAR, because the patches and patch scores are built from it. A valid predicted
  tier requires INDEPENDENT FoldX mutation predictions (PositionScan/BuildModel), a separate
  compute job. Not run here; the alanine-scan-as-outcome version is intentionally avoided.

## Central question - does more coverage reduce patch_score <-> RSA confounding?
These metrics are outcome-independent, so they are computed cleanly on the full
structural set (n=194) and compared with the experimental set (n=37):

| set | n | Spearman(patch,RSA) | Pearson(patch,RSA) | VIF |
|-----|--:|--------------------:|-------------------:|----:|
| experimental | {c37['n']} | {c37['spearman']:+.3f} | {c37['pearson']:+.3f} | {c37['vif']:.2f} |
| full structural | {c194['n']} | {c194['spearman']:+.3f} | {c194['pearson']:+.3f} | {c194['vif']:.2f} |

## Tier 1 (experimental) Q1 / Q2, unadjusted vs RSA-adjusted
- Q1 Fisher: OR = {orr.statistic:.2f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}], p = {pf:.4g}
- Q1 logistic patch OR: unadjusted = {np.exp(m1.params[1]):.2f} (p={m1.pvalues[1]:.3g});
  RSA-adjusted = {np.exp(m2.params[1]):.2f} (p={m2.pvalues[1]:.3g})
- Q2 Spearman(|rep ddG|, patch_score): rho = {rho:.2f}, p = {prho:.4g}
- Q2 OLS patch_score coef: unadjusted = {o1.params[1]:.4f} (p={o1.pvalues[1]:.3g});
  RSA-adjusted = {o2.params[1]:.4f} (p={o2.pvalues[1]:.3g})

## Conclusion
{verdict()}

## Notes / limitations
- Sample size is small; CIs are wide; analyses are exploratory.
- patch_score and RSA are strongly (negatively) correlated by construction - load-bearing
  core residues are buried - which is why the two cannot be separated here.
- A genuine independent predicted tier (FoldX PositionScan over unmeasured residues) and an
  orthogonal experimental proxy (e.g. a MaveDB VAMP-seq abundance dataset, analysed separately
  as function/abundance rather than ddG) are the sensible next steps if higher coverage is required.
"""
(RES / "REPORT_extended.md").write_text(report, encoding="utf-8")
print("\nSaved -> results/residue_table_full.csv, results/REPORT_extended.md")
