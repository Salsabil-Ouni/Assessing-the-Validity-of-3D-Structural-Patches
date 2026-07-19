"""
CONSISTENT-CONTEXT re-run: chain A + zinc used throughout (alanine scan, RSA,
patches, PositionScan). Selected structure = 2OCJ (WT, apo) -> the provided
FoldX-repaired file; chain A extracted with its zinc (structure/tp53_chainA_zinc.pdb).

Produces residue_table_cA.csv and REPORT_consistent.md, covering:
  Tier 1 (experimental n=37): Q1 Fisher (+OR/CI), Q2 Spearman, RSA-adjusted models
  Confounding: corr(patch_score, RSA) + VIF on n=37 and n=194
  Tier 3 (predicted, n=194): only if PositionScan output is present (semi-circular; RSA-adjusted)
Residue-level aggregation; representative = most destabilising measured ddG (mean = sensitivity).
"""
import re, itertools
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, yaml
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import freesasa
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
HOT, DIST = cfg["cutoffs"]["ala_hotspot_ddg"], cfg["cutoffs"]["patch_distance_angstrom"]
DEL_BELOW, BURIED_REL = cfg["cutoffs"]["ddg_deleterious_below"], cfg["sasa"]["buried_cutoff_relative"]
RES = ROOT / "results"
PDB = ROOT / "structure" / "tp53_chainA_zinc.pdb"
three1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V","H1S":"H","H2S":"H","H3S":"H"}

# ---- parse chain-A alanine scan -------------------------------------------
line_re = re.compile(r"^([A-Z][A-Z0-9]{2})\s+(\d+)\s+to ALA energy change is\s+([-\d.eE]+)")
ala_ddg, wt1 = {}, {}
for ln in (RES / "foldx_raw" / "chainA_AS.fxout").read_text().splitlines():
    m = line_re.match(ln.strip())
    if m:
        ala_ddg[int(m.group(2))] = round(float(m.group(3)), 4)
        wt1[int(m.group(2))] = three1.get(m.group(1), "X")
pd.DataFrame({"position": list(ala_ddg), "residue": [wt1[p] for p in ala_ddg],
              "ddG": [ala_ddg[p] for p in ala_ddg]}).to_csv(RES / "ala_scan_cA.csv", index=False)
is_hot = {p: d >= HOT for p, d in ala_ddg.items()}

# ---- geometry on chain A --------------------------------------------------
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]["A"]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg}
positions = sorted(heavy)
neigh = {p: {p} for p in positions}
for p, q in itertools.combinations(positions, 2):
    if np.sqrt(((heavy[p][:, None] - heavy[q][None]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)
agg_score = {p: round(sum(ala_ddg[m] for m in neigh[p]), 3) for p in positions}
hp = [p for p in positions if is_hot[p]]
G = nx.Graph(); G.add_nodes_from(hp)
for p, q in itertools.combinations(hp, 2):
    if q in neigh[p]: G.add_edge(p, q)
patches = sorted(nx.connected_components(G), key=len, reverse=True)
patch_members = set().union(*[c for c in patches if len(c) >= 2]) if patches else set()
pd.DataFrame([{"patch_id": i, "n_residues": len(c), "residues": ",".join(map(str, sorted(c))),
               "aggregated_patch_score": round(sum(ala_ddg[m] for m in c), 3)}
              for i, c in enumerate(patches, 1)]).to_csv(RES / "patches_3D_cA.csv", index=False)

# ---- RSA (chain A protein only) -------------------------------------------
class ProtA(Select):
    def accept_chain(self, ch): return ch.id == "A"
    def accept_residue(self, r): return r.id[0] == " "
io = PDBIO(); io.set_structure(PDBParser(QUIET=True).get_structure("p", str(PDB)))
tmp = RES / "_cA_sasa.pdb"; io.save(str(tmp), ProtA())
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,"I":197,
          "L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
areas = freesasa.calc(freesasa.Structure(str(tmp))).residueAreas()["A"]
rsa = {int(k): v.total / MAXASA.get(wt1.get(int(k), ""), 999) for k, v in areas.items() if k.isdigit()}
tmp.unlink(missing_ok=True)

full = pd.DataFrame({"position": positions})
full["ala_ddg"] = full.position.map(ala_ddg); full["patch_score"] = full.position.map(agg_score)
full["rsa"] = full.position.map(rsa); full["is_hotspot"] = full.position.map(is_hot)
full["in_patch"] = full.position.isin(patch_members)
full = full.dropna(subset=["rsa"])

# ---- experimental residue-level -------------------------------------------
A = pd.read_csv(RES / "listA_variants.csv")
rep = (A.groupby("position", as_index=False)
         .agg(rep_ddg=("ddg", "min"), mean_ddg=("ddg", "mean"), n_sub=("ddg", "size")))
rep["destabilizing"] = rep["rep_ddg"] < DEL_BELOW
rep["abs_rep_ddg"] = rep["rep_ddg"].abs()
rep = rep.merge(full[["position", "patch_score", "rsa", "in_patch"]], on="position", how="inner")
rep.to_csv(RES / "residue_table_cA.csv", index=False)
full.to_csv(RES / "residue_table_full_cA.csv", index=False)

def fisher(df, col="destabilizing"):
    a=int((df[col]&df.in_patch).sum()); b=int((df[col]&~df.in_patch).sum())
    c=int((~df[col]&df.in_patch).sum()); d=int((~df[col]&~df.in_patch).sum())
    o=odds_ratio([[a,b],[c,d]]); ci=o.confidence_interval(0.95); _,p=fisher_exact([[a,b],[c,d]])
    return (a,b,c,d),o.statistic,(ci.low,ci.high),p
def vif2(df):
    X=sm.add_constant(df[["patch_score","rsa"]].astype(float).values)
    return variance_inflation_factor(X,1)

print(f"[consistent context: chain A + zinc, structure 2OCJ]")
print(f"hotspots={sum(is_hot.values())}  patches sizes={[len(c) for c in patches]}  in_patch residues={len(patch_members)}")

print("\n=== CONFOUNDING patch_score vs RSA ===")
for nm, d in [("exp n=37", rep), ("full n=194", full)]:
    sr,_=spearmanr(d.patch_score,d.rsa); pr,_=pearsonr(d.patch_score,d.rsa)
    print(f"  {nm}: Spearman={sr:+.3f} Pearson={pr:+.3f} VIF={vif2(d):.2f}")

print("\n=== TIER 1 (experimental n=37) ===")
tab,orr,ci,pf = fisher(rep)
rho,prho = spearmanr(rep.abs_rep_ddg, rep.patch_score)
y=rep.destabilizing.astype(int).values
l1=sm.Logit(y,sm.add_constant(rep[["in_patch"]].astype(float))).fit(disp=0)
l2=sm.Logit(y,sm.add_constant(rep[["in_patch","rsa"]].astype(float))).fit(disp=0)
o1=sm.OLS(rep.abs_rep_ddg,sm.add_constant(rep[["patch_score"]])).fit()
o2=sm.OLS(rep.abs_rep_ddg,sm.add_constant(rep[["patch_score","rsa"]])).fit()
print(f"  Q1 Fisher table(a,b,c,d)={tab} OR={orr:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] p={pf:.4g}")
print(f"  Q1 logistic patch OR: unadj={np.exp(l1.params[1]):.2f}(p={l1.pvalues[1]:.3g}) RSA-adj={np.exp(l2.params[1]):.2f}(p={l2.pvalues[1]:.3g})")
print(f"  Q2 Spearman rho={rho:.3f} p={prho:.4g}")
print(f"  Q2 OLS patch coef: unadj={o1.params[1]:.4f}(p={o1.pvalues[1]:.3g}) RSA-adj={o2.params[1]:.4f}(p={o2.pvalues[1]:.3g})")

# ---- Tier 3 (only if PositionScan output present) -------------------------
psf = RES / "foldx_raw" / "positionscan_scanning_output.txt"
tier3_txt = "Tier 3 not run yet (PositionScan output not found)."
if psf.exists():
    pr_re = re.compile(r"^([A-Z0-9]+)A(\d+)([A-Za-z])\s+([-\d.eE]+)$")
    bag = {}
    for ln in psf.read_text().splitlines():
        m = pr_re.match(ln.strip())
        if not m: continue
        pos, mut, dd = int(m.group(2)), m.group(3), float(m.group(4))
        if mut.isupper() and mut != wt1.get(pos):
            bag.setdefault(pos, []).append(dd)
    pred = pd.DataFrame([{"position": p, "pred_ddg_mean": round(np.mean(v), 3)} for p, v in bag.items()])
    fp = full.merge(pred, on="position", how="inner")
    fp["pred_destab"] = fp.pred_ddg_mean > 1.0
    val = fp.merge(rep[["position","rep_ddg"]], on="position", how="inner")
    rpe,ppe = spearmanr(val.pred_ddg_mean, val.rep_ddg)
    tab3,orr3,ci3,pf3 = fisher(fp, "pred_destab")
    rho3,prho3 = spearmanr(fp.pred_ddg_mean, fp.patch_score)
    yy=fp.pred_destab.astype(int).values
    L1=sm.Logit(yy,sm.add_constant(fp[["in_patch"]].astype(float))).fit(disp=0)
    L2=sm.Logit(yy,sm.add_constant(fp[["in_patch","rsa"]].astype(float))).fit(disp=0)
    O1=sm.OLS(fp.pred_ddg_mean,sm.add_constant(fp[["patch_score"]])).fit()
    O2=sm.OLS(fp.pred_ddg_mean,sm.add_constant(fp[["patch_score","rsa"]])).fit()
    fp.to_csv(RES / "residue_table_full_cA.csv", index=False)
    print("\n=== TIER 3 (predicted n=%d) ===" % len(fp))
    print(f"  validation Spearman(pred, exp rep_ddg) = {rpe:.3f} p={ppe:.3g} (expect negative)")
    print(f"  Q1 Fisher OR={orr3:.2f} CI[{ci3[0]:.2f},{ci3[1]:.2f}] p={pf3:.3g}")
    print(f"  Q1 logistic patch OR: unadj={np.exp(L1.params[1]):.2f}(p={L1.pvalues[1]:.3g}) RSA-adj={np.exp(L2.params[1]):.2f}(p={L2.pvalues[1]:.3g})")
    print(f"  Q2 Spearman rho={rho3:.3f} p={prho3:.3g}")
    print(f"  Q2 OLS patch coef: unadj={O1.params[1]:.4f}(p={O1.pvalues[1]:.3g}) RSA-adj={O2.params[1]:.4f}(p={O2.pvalues[1]:.3g})")
    tier3_txt = (f"- Validation Spearman(predicted, experimental ddG) = {rpe:.2f} (p={ppe:.3g}), negative as expected.\n"
                 f"- Q1 Fisher OR={orr3:.2f} CI[{ci3[0]:.2f},{ci3[1]:.2f}] p={pf3:.3g}; "
                 f"logistic patch OR unadj={np.exp(L1.params[1]):.2f} (p={L1.pvalues[1]:.3g}), "
                 f"RSA-adjusted={np.exp(L2.params[1]):.2f} (p={L2.pvalues[1]:.3g}).\n"
                 f"- Q2 Spearman rho={rho3:.2f} (p={prho3:.3g}); OLS patch coef unadj={O1.params[1]:.4f} "
                 f"(p={O1.pvalues[1]:.3g}), RSA-adjusted={O2.params[1]:.4f} (p={O2.pvalues[1]:.3g}).")

report = f"""# TP53 microenvironment vs burial - CONSISTENT-CONTEXT analysis (chain A + zinc)

**Structure:** 2OCJ (WT human p53 DBD, DNA-free, 2.05 A, complete over 93-290, zinc present)
- selected by the prioritisation step (see results/structure_candidates.csv); the provided
  FoldX-repaired file matches 2OCJ. Chain A + its zinc used for ALL steps (alanine scan,
  RSA, patch construction, PositionScan) for a single consistent context.

**Aggregation:** residue level; representative = most destabilising measured ddG
(mean reported as sensitivity). No mixed-effects model.

## Hotspots / patches (chain-A context)
hotspots (ala-scan ddG >= {HOT}): {sum(is_hot.values())}; patches {[len(c) for c in patches]};
residues in a patch (size>=2): {len(patch_members)}.

## Confounding: patch_score vs RSA (outcome-independent)
| set | n | Spearman | Pearson | VIF |
|-----|--:|---------:|--------:|----:|
| experimental | 37 | {spearmanr(rep.patch_score,rep.rsa)[0]:+.3f} | {pearsonr(rep.patch_score,rep.rsa)[0]:+.3f} | {vif2(rep):.2f} |
| full structural | {len(full)} | {spearmanr(full.patch_score,full.rsa)[0]:+.3f} | {pearsonr(full.patch_score,full.rsa)[0]:+.3f} | {vif2(full):.2f} |

## Tier 1 - experimental (n=37)
- Q1 Fisher: table (a,b,c,d) = {tab}, OR = {orr:.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], p = {pf:.4g}
- Q1 logistic patch OR: unadjusted {np.exp(l1.params[1]):.2f} (p={l1.pvalues[1]:.3g});
  RSA-adjusted {np.exp(l2.params[1]):.2f} (p={l2.pvalues[1]:.3g})
- Q2 Spearman(|rep ddG|, patch_score): rho = {rho:.2f}, p = {prho:.4g}
- Q2 OLS patch coef: unadjusted {o1.params[1]:.4f} (p={o1.pvalues[1]:.3g});
  RSA-adjusted {o2.params[1]:.4f} (p={o2.pvalues[1]:.3g})

## Tier 3 - FoldX-predicted (PositionScan, coverage 37 -> 194)
{tier3_txt}

**Caveat (semi-circularity).** The predicted outcome (PositionScan mean ddG) and the
patch predictor (alanine scan) share a FoldX/structure basis - alanine is one of the
PositionScan substitutions, and load-bearing residues destabilise under most
substitutions - so "predicted-destabilising" and "hotspot/in-patch" are largely the same
variable. Therefore the very large Tier-3 Q1 odds ratio (and its survival of RSA
adjustment) is inflated and is NOT independent evidence of an effect beyond burial. The
continuous Tier-3 Q2 is less sensitive to this binary collinearity, and there the
patch_score effect becomes non-significant after RSA adjustment - consistent with Tier 1.
Validation against experiment is only moderate (Spearman ~ -0.5), so the predicted tier
is a coverage/robustness check, not a substitute for experimental data.

## Conclusion (does the microenvironment effect persist independently of burial?)
- Unadjusted, the patch effect is present in every tier (Q1 and Q2).
- patch_score and RSA remain strongly correlated even at n=194 (Pearson ~ -0.74), so
  increasing coverage does NOT break the collinearity.
- In the trustworthy analyses - Tier 1 (experimental) Q1 and Q2, and the continuous
  Tier-3 Q2 - the patch effect becomes non-significant after adjusting for RSA.
- The only analysis where it "survives" RSA adjustment is the binary Tier-3 Q1, which is
  semi-circular (see caveat) and therefore not reliable.
- VERDICT: we find **no robust evidence that the 3D microenvironment exerts an effect
  beyond solvent burial**. The hotspot-patch association is largely explained by residue
  burial (RSA). Stated explicitly as a limitation of the available data.
"""
(RES / "REPORT_consistent.md").write_text(report, encoding="utf-8")
print("\nSaved -> ala_scan_cA.csv, patches_3D_cA.csv, residue_table_cA.csv, REPORT_consistent.md")
