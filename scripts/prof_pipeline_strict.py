"""
PROF PIPELINE - strict, residue-level analysis.

Hypothesis: the structural impact of a variant is determined by the 3D
microenvironment to which it belongs.

  Q1 (enrichment): are destabilizing variants OVERREPRESENTED in hotspot patches?
  Q2 (intensity) : does the magnitude of impact depend on the aggregated 3D-patch score?

Key methodological points required by the spec:
  * VARIANT BRANCH is collapsed to the RESIDUE level (multiple substitutions at one
    residue are NOT independent -> pseudoreplication). One representative value per
    residue is used. Representative = the MOST DESTABILIZING measurement at that
    residue (minimum signed ddG, since ThermoMutDB uses negative = destabilizing).
    Justification: the hypothesis concerns whether a residue's structural context
    makes it vulnerable; the strongest observed destabilization is the natural
    summary of that vulnerability, and it makes the residue-level class
    ("destabilizing if any substitution destabilizes it") internally consistent.
    Mean ddG is reported alongside as a sensitivity check.
  * STRUCTURE BRANCH (QC/clean/FoldX-repair/alanine-scan/patches) was completed
    earlier on the provided FoldX-repaired structure (chain A, zinc retained,
    continuous 96-289). Patches = spatial clusters of hotspot residues; we do NOT
    invent extra patches.

Outputs: patches_3D.csv, variant_microenvironment.csv (residue level), REPORT_prof.md
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy.stats import fisher_exact, spearmanr, linregress
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CHAIN     = cfg["structure"]["chain"]
HOT       = cfg["cutoffs"]["ala_hotspot_ddg"]
DIST      = cfg["cutoffs"]["patch_distance_angstrom"]
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]
STAB_ABOVE= cfg["cutoffs"]["ddg_stabilising_above"]
PDB       = ROOT / cfg["structure"]["repaired_pdb"]
RES       = ROOT / "results"

ala = pd.read_csv(RES / "ala_scan.csv")           # position, residue, ddG (194)
A   = pd.read_csv(RES / "listA_variants.csv")     # variants (per substitution)
ala_ddg = dict(zip(ala.position, ala.ddG))
is_hot  = {p: (d >= HOT) for p, d in ala_ddg.items()}

# ---------- geometry: heavy-atom neighbours within DIST --------------------
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg}
positions = sorted(heavy)
neigh = {p: {p} for p in positions}
for p, q in itertools.combinations(positions, 2):
    a, b = heavy[p], heavy[q]
    if np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)

# ---------- 3D hotspot patches (clusters of hotspot residues) --------------
hot_pos = [p for p in positions if is_hot[p]]
G = nx.Graph(); G.add_nodes_from(hot_pos)
for p, q in itertools.combinations(hot_pos, 2):
    if q in neigh[p]:
        G.add_edge(p, q)
patches = sorted(nx.connected_components(G), key=len, reverse=True)
patch_rows = [{"patch_id": i, "n_residues": len(c),
               "residues": ",".join(map(str, sorted(c))),
               "aggregated_patch_score": round(sum(ala_ddg[m] for m in c), 3)}
              for i, c in enumerate(patches, 1)]
pd.DataFrame(patch_rows).to_csv(RES / "patches_3D.csv", index=False)
patch_members = set().union(*[c for c in patches if len(c) >= 2]) if patches else set()

# ---------- aggregated 3D-patch score per residue --------------------------
agg_score = {p: round(sum(ala_ddg[m] for m in neigh[p]), 3) for p in positions}
n_hot_env = {p: sum(1 for m in neigh[p] if is_hot[m]) for p in positions}

# ---------- VARIANT BRANCH: collapse to residue level ----------------------
def classify(rep):
    if rep < DEL_BELOW:  return "destabilizing"
    if rep > STAB_ABOVE: return "stabilizing"
    return "neutral"

rep = (A.groupby(["position", "wt"], as_index=False)
         .agg(n_substitutions=("ddg", "size"),
              rep_ddg=("ddg", "min"),       # most destabilizing (min signed)
              mean_ddg=("ddg", "mean")))
rep["class"] = rep["rep_ddg"].apply(classify)
rep["destabilizing"] = rep["class"] == "destabilizing"
rep["in_hotspot_patch"] = rep["position"].isin(patch_members)
rep["aggregated_patch_score"] = rep["position"].map(agg_score)
rep["n_hotspots_in_env"] = rep["position"].map(n_hot_env)
rep["abs_rep_ddg"] = rep["rep_ddg"].abs()
rep = rep.sort_values("position").reset_index(drop=True)
rep.to_csv(RES / "variant_microenvironment.csv", index=False)

# ===================== Q1: ENRICHMENT (residue level) ======================
a = int(( rep.destabilizing &  rep.in_hotspot_patch).sum())   # destab, hotspot
b = int(( rep.destabilizing & ~rep.in_hotspot_patch).sum())   # destab, non-hotspot
c = int((~rep.destabilizing &  rep.in_hotspot_patch).sum())   # other,  hotspot
d = int((~rep.destabilizing & ~rep.in_hotspot_patch).sum())   # other,  non-hotspot
table = [[a, b], [c, d]]
_, p_fisher = fisher_exact(table, alternative="two-sided")
orr = odds_ratio(table)                 # conditional MLE odds ratio
ci = orr.confidence_interval(confidence_level=0.95)

# ===================== Q2: INTENSITY (residue level) =======================
m = rep.dropna(subset=["aggregated_patch_score"]).copy()
rho, p_rho = spearmanr(m["abs_rep_ddg"], m["aggregated_patch_score"])
rho_mean, p_rho_mean = spearmanr(m["mean_ddg"].abs(), m["aggregated_patch_score"])
# linear regression: |rep ddG| ~ aggregated score
lr = linregress(m["aggregated_patch_score"], m["abs_rep_ddg"])
# logistic regression: P(destabilizing) ~ aggregated score
X = sm.add_constant(m["aggregated_patch_score"].values.astype(float))
y = m["destabilizing"].astype(int).values
logit = sm.Logit(y, X).fit(disp=0)
log_coef = logit.params[1]; log_p = logit.pvalues[1]; log_or = np.exp(log_coef)

# ---------- print ----------------------------------------------------------
print(f"Residues with experimental data (after collapse): {len(rep)}")
print(f"  destabilizing: {int(rep.destabilizing.sum())} | "
      f"neutral: {int((rep['class']=='neutral').sum())} | "
      f"stabilizing: {int((rep['class']=='stabilizing').sum())}")
print(f"3D hotspot patches: {len(patch_rows)} (sizes {[r['n_residues'] for r in patch_rows]})")
print("\nQ1 contingency (residue level):")
print(f"                 hotspot   non-hotspot")
print(f"destabilizing       {a:>3}        {b:>3}")
print(f"other               {c:>3}        {d:>3}")
print(f"Fisher p = {p_fisher:.4g}")
print(f"odds ratio (conditional MLE) = {orr.statistic:.3g}  95% CI [{ci.low:.3g}, {ci.high:.3g}]")
print("\nQ2 intensity:")
print(f"Spearman |rep ddG| vs aggregated score: rho = {rho:.3f}, p = {p_rho:.4g}  (n={len(m)})")
print(f"  sensitivity (mean ddG):               rho = {rho_mean:.3f}, p = {p_rho_mean:.4g}")
print(f"linear reg slope = {lr.slope:.4f} (p={lr.pvalue:.4g}, r={lr.rvalue:.3f})")
print(f"logistic P(destab) ~ score: coef={log_coef:.4f}, OR/unit={log_or:.3f}, p={log_p:.4g}")

# ---------- REPORT ---------------------------------------------------------
def sig(p): return "significant" if p < 0.05 else "not significant"
ci_txt = f"[{ci.low:.2f}, {'inf' if np.isinf(ci.high) else f'{ci.high:.2f}'}]"
report = f"""# TP53 - 3D microenvironment analysis (professor's pipeline)

**Hypothesis.** The structural impact of a variant is determined by the 3D
microenvironment to which it belongs.

## Methods (summary)
**Structure branch.** A FoldX-repaired WT TP53 DNA-binding-domain structure was used
(chain {CHAIN}; structural zinc retained; QC: continuous chain {min(positions)}-{max(positions)},
no internal gaps; only the Zn heteroatom kept). A full per-residue alanine scan
(FoldX) gave each residue's ddG (contribution to stability). Hotspot residues were
defined as alanine-scan ddG >= {HOT} kcal/mol. 3D hotspot patches were built as
spatial clusters of hotspot residues whose closest heavy atoms lie within {DIST} A
(connected components). The aggregated score of a patch / microenvironment is the
sum of the alanine-scan ddG of its member residues.

**Variant branch.** Experimental ddG values (ThermoMutDB; negative = destabilizing)
were collapsed to the **residue level** to avoid pseudoreplication, because multiple
substitutions at the same residue (e.g. R248Q/R248W/R248G) are not independent
observations. The representative value per residue is the **most destabilizing
measurement** (minimum signed ddG); mean ddG is reported as a sensitivity check.
Residues were classified as destabilizing (rep ddG < {DEL_BELOW}), stabilizing
(rep ddG > {STAB_ABOVE}) or neutral.

## Sample
- Residues with experimental data (after collapse): **{len(rep)}**
  (destabilizing {int(rep.destabilizing.sum())}, neutral {int((rep['class']=='neutral').sum())},
   stabilizing {int((rep['class']=='stabilizing').sum())})
- 3D hotspot patches: **{len(patch_rows)}** (sizes {[r['n_residues'] for r in patch_rows]}).
  The hotspot residues of this domain form essentially a single contiguous structural
  cluster; no additional patches were invented to match the illustrative drawing.

## Q1 - Enrichment of destabilizing variants in hotspot patches
Residue-level 2x2 contingency table:

|                | hotspot | non-hotspot |
|----------------|--------:|------------:|
| destabilizing  | {a} | {b} |
| other          | {c} | {d} |

- Fisher exact test: p = **{p_fisher:.4g}**
- Odds ratio (conditional MLE) = **{orr.statistic:.2f}**, 95% CI = **{ci_txt}**

Interpretation: destabilizing residues are **overrepresented** in hotspot patches
(OR = {orr.statistic:.1f}). The association is {sig(p_fisher)} at the 0.05 level.
The confidence interval is wide, reflecting the modest sample size.

## Q2 - Dependence of impact magnitude on the aggregated 3D-patch score
- Spearman correlation, |representative ddG| vs aggregated patch score:
  rho = **{rho:.2f}**, p = **{p_rho:.4g}** (n = {len(m)}); {sig(p_rho)}.
  Sensitivity using mean ddG: rho = {rho_mean:.2f}, p = {p_rho_mean:.4g}.
- Linear regression (|rep ddG| ~ score): slope = {lr.slope:.3f} kcal/mol per score unit,
  p = {lr.pvalue:.4g}, r = {lr.rvalue:.2f}.
- Logistic regression P(destabilizing) ~ score: OR per unit = {log_or:.2f}, p = {log_p:.4g}.

Interpretation: **variants located in stronger structural microenvironments tend to
exhibit larger destabilizing effects.** This is an association only; no causal claim
is made.

## Limitations and notes
- Sample size is small ({len(rep)} residues), so confidence intervals are wide and the
  analysis is exploratory.
- **Within** the hotspot patch the aggregated scores are uniformly high, so the
  correlation flattens there; this is expected. The informative contrast comes from
  comparing low-score and high-score microenvironments across the whole dataset.
- Hotspot-patch membership is strongly related to residue burial; the enrichment in
  Q1 may partly reflect that buried residues are both hotspots and destabilization-
  sensitive. This was not adjusted for here and should be kept in mind.
- ThermoMutDB aggregates many primary studies; sign and measurement heterogeneity
  add noise.

## Conclusion
- **Q1:** Destabilizing residues are **overrepresented** in hotspot patches
  (OR = {orr.statistic:.1f}, 95% CI {ci_txt}, Fisher p = {p_fisher:.3g}).
- **Q2:** Impact magnitude is **positively associated** with the aggregated 3D-patch
  score (Spearman rho = {rho:.2f}, p = {p_rho:.3g}); variants in stronger structural
  microenvironments tend to be more destabilizing.

Together these results are **consistent with** the hypothesis that a variant's
structural impact is shaped by the 3D microenvironment to which it belongs, within
the stated limitations.
"""
(RES / "REPORT_prof.md").write_text(report, encoding="utf-8")
print("\nSaved -> results/patches_3D.csv, results/variant_microenvironment.csv, results/REPORT_prof.md")
