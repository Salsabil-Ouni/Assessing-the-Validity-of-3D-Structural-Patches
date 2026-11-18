"""
Multi-structure validation, part B: repeat Q1/Q2 on 2XWR (chain B + zinc,
FoldX-repaired) exactly as done for 2OCJ in consistent_context.py, and compare.

This answers the supervisor's own open question on the pipeline sketch ("are we
going to use all the P53 WT structures or not") that the main analysis (built on
2OCJ alone) never revisited: does the hotspot-patch enrichment/intensity finding
reproduce on a second, independent WT p53 structure?
"""
import re
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import yaml
import freesasa
from Bio.PDB import PDBParser, PDBIO, Select
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
HOT = cfg["cutoffs"]["ala_hotspot_ddg"]
DIST = 6.0  # Ca-Ca, matching the supervisor's hotspotPatches.py (primary patch definition)
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]
PDB = ROOT / "structure" / "tp53_2xwr_chainB_zinc_Repair.pdb"

three1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","H1S":"H","H2S":"H","H3S":"H"}

# ---- parse AlaScan output ---------------------------------------------------
line_re = re.compile(r"^([A-Z][A-Z0-9]{2})\s+(\d+)\s+to ALA energy change is\s+([-\d.eE]+)")
ala_ddg, wt1 = {}, {}
for ln in (RES / "foldx_raw" / "2xwr_chainB_AS.fxout").read_text().splitlines():
    m = line_re.match(ln.strip())
    if m:
        ala_ddg[int(m.group(2))] = round(float(m.group(3)), 4)
        wt1[int(m.group(2))] = three1.get(m.group(1), "X")
is_hot = {p: d >= HOT for p, d in ala_ddg.items()}
print(f"2XWR chain B: {len(ala_ddg)} residues profiled; {sum(is_hot.values())} hotspots (g_i >= {HOT})")

# sanity check: are the zinc ligands strong hotspots, as in 2OCJ?
zn_lig = [176, 179, 238, 242]
top4 = sorted(ala_ddg, key=lambda p: -ala_ddg[p])[:4]
zn_all_hotspots = all(is_hot.get(p, False) for p in zn_lig)
print(f"Zinc ligand ala-scan values: {[(p, ala_ddg[p]) for p in zn_lig]} "
      f"-> all flagged as hotspots: {zn_all_hotspots}")
print(f"Top-4 by magnitude: {[(p, ala_ddg[p]) for p in top4]} "
      f"(exact match to zinc ligands: {set(top4) == set(zn_lig)}; "
      f"in 2OCJ the top-4 were exactly the 4 zinc ligands)")

# ---- geometry: Ca-Ca neighbours + patches -----------------------------------
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]["B"]
heavy = {r.id[1]: r["CA"].coord for r in chain if r.id[1] in ala_ddg and r.has_id("CA")}
positions = sorted(heavy)
neigh = {p: {p} for p in positions}
for p, q in itertools.combinations(positions, 2):
    if np.linalg.norm(heavy[p] - heavy[q]) <= DIST:
        neigh[p].add(q); neigh[q].add(p)
agg_score = {p: round(sum(ala_ddg[m] for m in neigh[p]), 3) for p in positions}
hp = [p for p in positions if is_hot[p]]
G = nx.Graph(); G.add_nodes_from(hp)
for p, q in itertools.combinations(hp, 2):
    if q in neigh[p]:
        G.add_edge(p, q)
patches = sorted(nx.connected_components(G), key=len, reverse=True)
patch_members = set().union(*[c for c in patches if len(c) >= 2]) if patches else set()
print(f"Patches (size>=2): {[len(c) for c in patches if len(c) >= 2]}  "
      f"(2OCJ found a single 63-residue core + 1 isolated hotspot)")

# ---- RSA ---------------------------------------------------------------------
class ProtB(Select):
    def accept_chain(self, ch):
        return ch.id == "B"
    def accept_residue(self, r):
        return r.id[0] == " "
io = PDBIO(); io.set_structure(PDBParser(QUIET=True).get_structure("p", str(PDB)))
tmp = RES / "_2xwr_sasa.pdb"; io.save(str(tmp), ProtB())
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,"I":197,
          "L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
areas = freesasa.calc(freesasa.Structure(str(tmp))).residueAreas()["B"]
rsa = {int(k): v.total / MAXASA.get(wt1.get(int(k), ""), 999) for k, v in areas.items() if k.isdigit()}
tmp.unlink(missing_ok=True)

full = pd.DataFrame({"position": positions})
full["ala_ddg"] = full.position.map(ala_ddg); full["patch_score"] = full.position.map(agg_score)
full["rsa"] = full.position.map(rsa); full["is_hotspot"] = full.position.map(is_hot)
full["in_patch"] = full.position.isin(patch_members)
full = full.dropna(subset=["rsa"])
full.to_csv(RES / "residue_table_2xwr.csv", index=False)

# ---- merge with the SAME 37-residue experimental set used for 2OCJ ----------
A = pd.read_csv(RES / "listA_variants.csv")
rep = (A.groupby("position", as_index=False).agg(rep_ddg=("ddg", "min")))
rep["destabilizing"] = rep["rep_ddg"] < DEL_BELOW
rep["abs_rep_ddg"] = rep["rep_ddg"].abs()
n_before = len(rep)
rep = rep.merge(full[["position", "patch_score", "rsa", "in_patch"]], on="position", how="inner")
print(f"\nExperimental positions: {n_before} total, {len(rep)} covered by the 2XWR construct (89-293)")

# wt-identity check on the merged residues
A_wt = dict(zip(A.position, A.wt))
mismatches = [(p, A_wt[p], wt1.get(p)) for p in rep.position if A_wt.get(p) != wt1.get(p)]
print(f"wt-identity mismatches on merged residues: {len(mismatches)} {mismatches if mismatches else '(none)'}")

def fisher(df):
    a = int((df.destabilizing & df.in_patch).sum()); b = int((df.destabilizing & ~df.in_patch).sum())
    c = int((~df.destabilizing & df.in_patch).sum()); d = int((~df.destabilizing & ~df.in_patch).sum())
    o = odds_ratio([[a, b], [c, d]]); ci = o.confidence_interval(0.95)
    _, p = fisher_exact([[a, b], [c, d]])
    return (a, b, c, d), o.statistic, (ci.low, ci.high), p

tab, orr, ci, pf = fisher(rep)
rho, prho = spearmanr(rep.abs_rep_ddg, rep.patch_score)

print(f"\n=== 2XWR validation (n={len(rep)}) ===")
print(f"  Q1 Fisher: table={tab} OR={orr:.2f} 95%CI[{ci[0]:.2f},{ci[1]:.2f}] p={pf:.4g}")
print(f"  Q2 Spearman(|ddG|, patch_score): rho={rho:.3f} p={prho:.4g}")
print(f"\n  For comparison, 2OCJ (main text): Q1 OR=8.0 [1.5,59] p=0.007; Q2 rho=0.51 p=0.001")

# cross-structure consistency: do the two structures' own ala-scan profiles agree?
ocj_ala = pd.read_csv(RES / "ala_scan_cA.csv")[["position", "ddG"]].rename(columns={"ddG": "ala_ddg_2ocj"})
cmp = full[["position", "ala_ddg"]].rename(columns={"ala_ddg": "ala_ddg_2xwr"}).merge(ocj_ala, on="position")
r_cross, p_cross = spearmanr(cmp.ala_ddg_2xwr, cmp.ala_ddg_2ocj)
print(f"\n  Cross-structure profile agreement (n={len(cmp)} shared positions):")
print(f"  Spearman(ala_ddg_2XWR, ala_ddg_2OCJ) = {r_cross:.3f} (p={p_cross:.3g})")

pd.DataFrame([{
    "structure": "2XWR", "n": len(rep), "n_hotspots": sum(is_hot.values()),
    "n_patches_ge2": len([c for c in patches if len(c) >= 2]),
    "Q1_OR": round(orr, 2), "Q1_CI_low": round(ci[0], 2), "Q1_CI_high": round(ci[1], 2), "Q1_p": round(pf, 5),
    "Q2_rho": round(rho, 3), "Q2_p": round(prho, 5),
    "cross_structure_ala_ddg_rho": round(r_cross, 3), "cross_structure_ala_ddg_p": round(p_cross, 5),
    "n_wt_mismatches": len(mismatches),
}]).to_csv(RES / "second_structure_validation.csv", index=False)
print("\nSaved -> results/residue_table_2xwr.csv, results/second_structure_validation.csv")
