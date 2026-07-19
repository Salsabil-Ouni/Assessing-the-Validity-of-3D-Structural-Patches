"""
Sensitivity of the result to the neighbourhood / patch distance cut-off.
Sweep D = 4, 5, 6, 8 A and, at each D, recompute:
  - patch structure (does it fragment into >1 patch?)
  - Q1 enrichment (Fisher, full experimental set)
  - Q2 intensity (Spearman, full set)
  - Q2 within BURIED residues only (RSA<0.1)  <- the 'beyond burial' test
  - correlation between microenvironment score and RSA (does a tighter radius
    de-couple the score from burial?)
Report ALL distances (no cherry-picking).
"""
import itertools
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, yaml
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats.contingency import odds_ratio
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
HOT = cfg["cutoffs"]["ala_hotspot_ddg"]
PDB = ROOT / "structure" / "tp53_chainA_zinc.pdb"
RSA_CUT = 0.10

ala  = pd.read_csv(ROOT / "results" / "ala_scan_cA.csv")
exp  = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")       # 37: rep_ddg, abs_rep_ddg, destabilizing, rsa
full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")  # 194: rsa
ala_ddg = dict(zip(ala.position, ala.ddG))
rsa_all = dict(zip(full.position, full.rsa))

chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]["A"]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg}
pos = sorted(heavy)
# all pairwise min heavy-atom distances once
mind = {}
for p, q in itertools.combinations(pos, 2):
    mind[(p, q)] = np.sqrt(((heavy[p][:, None] - heavy[q][None]) ** 2).sum(-1)).min()

def within(p, q, D):
    a, b = (p, q) if p < q else (q, p)
    return mind[(a, b)] <= D

print(f"{'D(A)':>4} | {'patches(>=2)':>12} | {'Q1 OR (p)':>16} | {'Q2 full rho(p)':>16} | {'Q2 buried rho(p)':>17} | {'corr(score,RSA)':>15}")
print("-"*95)
for D in [2.5, 3.0, 3.5, 4.0]:
    # microenvironment score (include self)
    score = {}
    for p in pos:
        s = ala_ddg[p]
        for q in pos:
            if q != p and within(p, q, D):
                s += ala_ddg[q]
        score[p] = s
    # patches among hotspots
    hp = [p for p in pos if ala_ddg[p] >= HOT]
    G = nx.Graph(); G.add_nodes_from(hp)
    for p, q in itertools.combinations(hp, 2):
        if within(p, q, D): G.add_edge(p, q)
    comps = sorted((c for c in nx.connected_components(G)), key=len, reverse=True)
    big = [len(c) for c in comps if len(c) >= 2]
    members = set().union(*[c for c in comps if len(c) >= 2]) if any(len(c) >= 2 for c in comps) else set()

    e = exp.copy()
    e["score"] = e.position.map(score); e["inp"] = e.position.isin(members)
    # Q1
    a=int((e.destabilizing&e.inp).sum()); b=int((e.destabilizing&~e.inp).sum())
    c=int((~e.destabilizing&e.inp).sum()); d=int((~e.destabilizing&~e.inp).sum())
    if min(a+b,c+d,a+c,b+d)==0:
        q1=f"n/a{ (a,b,c,d) }"
    else:
        o=odds_ratio([[a,b],[c,d]]); _,pf=fisher_exact([[a,b],[c,d]]); q1=f"{o.statistic:.1f} (p={pf:.3f})"
    # Q2 full + buried
    rho,pr=spearmanr(e.abs_rep_ddg, e.score)
    eb=e[e.rsa<RSA_CUT]
    rb,pb=spearmanr(eb.abs_rep_ddg, eb.score)
    # corr(score, RSA) over all 194
    sc194=np.array([score[p] for p in pos]); rs194=np.array([rsa_all[p] for p in pos])
    csr,_=spearmanr(sc194, rs194)
    print(f"{D:>4.1f} | {str(big):>12} | {q1:>16} | {rho:>+7.2f}(p={pr:.3f}) | {rb:>+7.2f}(p={pb:.3f}) | {csr:>+15.2f}")
print("\nBURIED-only test is the 'beyond burial' one (Q2 buried). n(buried, RSA<0.1)="
      f"{int((exp.rsa<RSA_CUT).sum())}")
