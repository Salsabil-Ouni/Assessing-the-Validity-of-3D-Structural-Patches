"""
Burial-adjusted sensitivity model (supplement).

Question: does the hotspot-patch / microenvironment signal survive after adjusting
for residue burial (relative solvent accessibility)? Hotspot residues tend to be
buried, and buried residues tend to be destabilization-sensitive, so burial is a
candidate confounder.

We add relative SASA (freesasa, Tien 2013 max-ASA reference) to the residue-level
table and fit:
  Q1  logistic: P(destabilizing) ~ in_hotspot_patch              (unadjusted)
      logistic: P(destabilizing) ~ in_hotspot_patch + rel_sasa   (burial-adjusted)
  Q2  OLS:      |rep ddG| ~ aggregated_patch_score               (unadjusted)
      OLS:      |rep ddG| ~ aggregated_patch_score + rel_sasa    (burial-adjusted)
Small n -> exploratory; perfect-separation issues are reported, not hidden.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
import freesasa
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CHAIN = cfg["structure"]["chain"]
BURIED_REL = cfg["sasa"]["buried_cutoff_relative"]
PDB = ROOT / cfg["structure"]["repaired_pdb"]
RES = ROOT / "results"

rep = pd.read_csv(RES / "variant_microenvironment.csv")
ala = pd.read_csv(RES / "ala_scan.csv")

# ---- relative SASA per residue (chain A monomer) --------------------------
class ChainAProt(Select):
    def accept_chain(self, ch): return ch.id == CHAIN
    def accept_residue(self, r): return r.id[0] == " "
io = PDBIO(); io.set_structure(PDBParser(QUIET=True).get_structure("p", str(PDB)))
tmp = RES / "_chainA_sasa.pdb"; io.save(str(tmp), ChainAProt())
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,
          "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
res_wt = dict(zip(ala.position, ala.residue))
areas = freesasa.calc(freesasa.Structure(str(tmp))).residueAreas()[CHAIN]
relsasa = {int(k): v.total / MAXASA.get(res_wt.get(int(k), ""), 999)
           for k, v in areas.items() if k.isdigit()}
tmp.unlink(missing_ok=True)

rep["rel_sasa"] = rep["position"].map(relsasa)
rep["buried"] = rep["rel_sasa"] < BURIED_REL
y = rep["destabilizing"].astype(int).values

def fit_logit(cols, label):
    X = sm.add_constant(rep[cols].astype(float).values)
    try:
        m = sm.Logit(y, X).fit(disp=0)
        out = []
        for i, c in enumerate(["const"] + cols):
            out.append(f"    {c:>22}: coef={m.params[i]:+.3f}  OR={np.exp(m.params[i]):.2f}  p={m.pvalues[i]:.3g}")
        print(f"  [{label}]  (pseudo-R2={m.prsquared:.2f})")
        print("\n".join(out))
    except Exception as e:
        print(f"  [{label}] could not fit by MLE: {type(e).__name__} - "
              f"likely perfect separation (patch and burial too collinear at n={len(rep)}).")

def fit_ols(cols, label):
    X = sm.add_constant(rep[cols].astype(float).values)
    m = sm.OLS(rep["abs_rep_ddg"].values, X).fit()
    print(f"  [{label}]  (R2={m.rsquared:.2f})")
    for i, c in enumerate(["const"] + cols):
        print(f"    {c:>22}: coef={m.params[i]:+.4f}  p={m.pvalues[i]:.3g}")

print(f"Residues: {len(rep)}  buried: {int(rep.buried.sum())}  exposed: {int((~rep.buried).sum())}")
print(f"in_hotspot_patch vs buried cross-tab:")
print(pd.crosstab(rep.buried, rep.in_hotspot_patch).to_string())

print("\n=== Q1: P(destabilizing) logistic ===")
fit_logit(["in_hotspot_patch"], "unadjusted")
fit_logit(["in_hotspot_patch", "rel_sasa"], "burial-adjusted")

print("\n=== Q2: |rep ddG| linear (OLS) ===")
fit_ols(["aggregated_patch_score"], "unadjusted")
fit_ols(["aggregated_patch_score", "rel_sasa"], "burial-adjusted")

rep.to_csv(RES / "variant_microenvironment.csv", index=False)  # add sasa columns
print("\nUpdated results/variant_microenvironment.csv with rel_sasa, buried columns.")
