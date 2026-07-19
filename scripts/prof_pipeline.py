"""
PROF-ALIGNED statistical analysis.

Hypothesis (prof): "the impact of a variant is determined by the MICROENVIRONMENT
to which it belongs."

Patch definition (prof-aligned):
  Every residue belongs to a 3D MICROENVIRONMENT = all chain-A residues whose
  closest heavy atom is within `patch_distance_angstrom`. The microenvironment's
  AGGREGATED 3D-PATCH SCORE = sum of the alanine-scan ddG of its member residues
  (the local "load-bearing-ness" around that residue). A residue is "in a hotspot
  patch region" if its microenvironment contains at least one hotspot
  (alanine-scan ddG >= ala_hotspot_ddg).

Two questions (prof):
  Q1 DENSITY  : are DESTABILISING variants denser in hotspot-patch regions?
                -> variant-level 2x2 Fisher exact test.
  Q2 INTENSITY: does the size of a variant's impact depend on the aggregated
                3D-patch score of its microenvironment?
                -> variant-level Spearman of |exp ddG| vs aggregated patch score.

Outputs:
  results/patches_3D.csv             list of hotspot 3D patches (clusters) + score
  results/variant_microenvironment.csv  per-variant patch score + membership
  results/REPORT_prof.md             the two questions answered in plain English
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy.stats import fisher_exact, spearmanr
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CHAIN   = cfg["structure"]["chain"]
HOT     = cfg["cutoffs"]["ala_hotspot_ddg"]
DIST    = cfg["cutoffs"]["patch_distance_angstrom"]
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]
PDB     = ROOT / cfg["structure"]["repaired_pdb"]
RES     = ROOT / "results"

ala = pd.read_csv(RES / "ala_scan.csv")          # position, residue, ddG (all 194)
A   = pd.read_csv(RES / "listA_variants.csv")    # variants: mutation, position, ddg, label
ala_ddg = dict(zip(ala.position, ala.ddG))
is_hot  = {p: (d >= HOT) for p, d in ala_ddg.items()}

# ---- heavy-atom coordinates for every chain-A residue ---------------------
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
heavy = {}
for r in chain:
    p = r.id[1]
    if p in ala_ddg:
        coords = [a.coord for a in r if (a.element or "").strip().upper() != "H"]
        if coords:
            heavy[p] = np.array(coords)
positions = sorted(heavy)

# ---- pairwise minimum heavy-atom distances -> neighbour sets --------------
neigh = {p: {p} for p in positions}             # each residue is in its own microenvironment
for p, q in itertools.combinations(positions, 2):
    a, b = heavy[p], heavy[q]
    if np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)

# ---- list of hotspot 3D patches (clusters of hotspot residues) ------------
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

# residues that ARE part of a hotspot patch (cluster of size >= 2)
patch_members = set().union(*[c for c in patches if len(c) >= 2]) if patches else set()

# ---- per-residue microenvironment: aggregated patch score + membership ----
env = []
for p in positions:
    members = neigh[p]
    score = sum(ala_ddg[m] for m in members)             # aggregated 3D-patch score
    n_hot = sum(1 for m in members if is_hot[m])
    env.append({"position": p,
                "n_env_residues": len(members),
                "n_hotspots_in_env": n_hot,
                "aggregated_patch_score": round(score, 3),
                # Q1 membership: the residue itself sits ON a hotspot patch
                "in_hotspot_patch": p in patch_members})
envdf = pd.DataFrame(env)

# ---- attach microenvironment info to each VARIANT -------------------------
v = A.merge(envdf, on="position", how="left")
v["destabilising"] = v["label"] == "deleterious"
v["abs_ddg"] = v["ddg"].abs()
v.to_csv(RES / "variant_microenvironment.csv", index=False)

# ===================== Q1: DENSITY (variant-level Fisher) ==================
a = int((v.destabilising & v.in_hotspot_patch).sum())
b = int((v.destabilising & ~v.in_hotspot_patch).sum())
c = int((~v.destabilising & v.in_hotspot_patch).sum())
d = int((~v.destabilising & ~v.in_hotspot_patch).sum())
or_, p_two = fisher_exact([[a, b], [c, d]], alternative="two-sided")
_,   p_grt = fisher_exact([[a, b], [c, d]], alternative="greater")

# ===================== Q2: INTENSITY (variant-level Spearman) ==============
rho_all, prho_all = spearmanr(v["abs_ddg"], v["aggregated_patch_score"])
vin = v[v.in_hotspot_patch]
rho_in, prho_in = (spearmanr(vin["abs_ddg"], vin["aggregated_patch_score"])
                   if len(vin) >= 3 else (float("nan"), float("nan")))

# ---- print ----------------------------------------------------------------
print(f"Variants analysed: {len(v)}  (destabilising: {int(v.destabilising.sum())})")
print(f"3D hotspot patches (clusters): {len(patch_rows)}  sizes={[r['n_residues'] for r in patch_rows]}")
print("\n=== Q1 DENSITY (variant level) ===")
print(f"                in_patch  not_in_patch")
print(f"destabilising      {a:>3}       {b:>3}")
print(f"other              {c:>3}       {d:>3}")
print(f"odds ratio={or_:.3g}  p(two-sided)={p_two:.4g}  p(greater)={p_grt:.4g}")
print("\n=== Q2 INTENSITY (variant level): |exp ddG| vs aggregated 3D-patch score ===")
print(f"all variants (n={len(v)}):     rho={rho_all:.3f}  p={prho_all:.4g}")
print(f"in-patch variants (n={len(vin)}): rho={rho_in:.3f}  p={prho_in:.4g}")
print(f"\naggregated_patch_score range: {v.aggregated_patch_score.min():.1f} .. {v.aggregated_patch_score.max():.1f}")

# ---- REPORT ---------------------------------------------------------------
def sig(p): return "statistically significant" if p < 0.05 else "NOT statistically significant"
report = f"""# p53 - PROF-ALIGNED analysis: does a variant's impact depend on its 3D microenvironment?

**Hypothesis:** the impact of a variant is determined by the microenvironment
(local 3D patch) to which it belongs.

**Patch / microenvironment definition.** Each residue's microenvironment = all
chain-{CHAIN} residues whose closest heavy atom is within {DIST} A (includes the
residue itself). Its **aggregated 3D-patch score** = sum of the alanine-scan ddG
of those residues. For Q1, a variant is "in a hotspot-patch region" if its own
residue is a member of a hotspot 3D patch (a spatial cluster of hotspot residues,
each with alanine-scan ddG >= {HOT}); for Q2 we use the continuous aggregated
3D-patch score of its microenvironment.

- Variants analysed: **{len(v)}** (destabilising: **{int(v.destabilising.sum())}**)
- Hotspot 3D patches (spatial clusters of hotspots): **{len(patch_rows)}**
  (sizes {[r['n_residues'] for r in patch_rows]}) - see patches_3D.csv

## Q1 - DENSITY: are destabilising variants denser in hotspot-patch regions?
Variant-level 2x2 (Fisher exact):

|                | in hotspot patch | not in patch |
|----------------|-----------------:|-------------:|
| destabilising  | {a} | {b} |
| other          | {c} | {d} |

- Odds ratio = **{or_:.3g}**, p(two-sided) = **{p_two:.4g}**, p(one-sided enrichment) = **{p_grt:.4g}**

Plain English: destabilising variants are {'more' if or_>1 else 'not more'} concentrated
in hotspot-patch regions; the enrichment is {sig(p_grt)} (one-sided p = {p_grt:.3g}).

## Q2 - INTENSITY: does impact depend on the aggregated 3D-patch score?
Variant-level Spearman correlation of |experimental ddG| against the aggregated
3D-patch score of each variant's microenvironment:

- All variants (n={len(v)}):       rho = **{rho_all:.3f}**, p = **{prho_all:.4g}**
- In-patch variants (n={len(vin)}): rho = **{rho_in:.3f}**, p = **{prho_in:.4g}**

Plain English: a positive rho means variants sitting in 3D patches with a higher
aggregated alanine-scan score tend to be more destabilising. The all-variant
correlation rho = {rho_all:.2f} is {sig(prho_all)}.

## Cutoffs used (config.yaml)
- Region: {cfg['protein']['region_start']}-{cfg['protein']['region_end']}
- Destabilising variant: experimental ddG < {DEL_BELOW} kcal/mol (ThermoMutDB sign)
- Hotspot: alanine-scan ddG >= {HOT} kcal/mol
- Microenvironment / patch distance: closest heavy atoms within {DIST} A
- Structure: chain {CHAIN} of {cfg['structure']['repaired_pdb']} (zinc present)
"""
(RES / "REPORT_prof.md").write_text(report, encoding="utf-8")
print("\nSaved -> results/patches_3D.csv, results/variant_microenvironment.csv, results/REPORT_prof.md")
