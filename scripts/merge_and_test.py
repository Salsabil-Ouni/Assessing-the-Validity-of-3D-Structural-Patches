"""
MERGE + HYPOTHESIS TEST  (steps M2-M6).

Unit of analysis = RESIDUE (position). A position is 'deleterious' if it has at
least one destabilising mutation in List A (its most-destabilising ddG < cutoff).
The test universe = positions that have experimental data (List A).

M2  merge List A (per residue) with patch membership + alanine-scan ddG
M3  2x2 Fisher exact test: deleterious vs in_patch
M4  intensity: Spearman of experimental |ddG| vs each residue's alanine-scan ddG
M5  confounder: buried vs exposed (freesasa), repeat Fisher within each stratum
M6  write results/REPORT.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import fisher_exact, spearmanr
import freesasa
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DEL_BELOW   = cfg["cutoffs"]["ddg_deleterious_below"]   # ddG < this  -> destabilising
STAB_ABOVE  = cfg["cutoffs"]["ddg_stabilising_above"]
HOT         = cfg["cutoffs"]["ala_hotspot_ddg"]
DIST        = cfg["cutoffs"]["patch_distance_angstrom"]
BURIED_REL  = cfg["sasa"]["buried_cutoff_relative"]
CHAIN       = cfg["structure"]["chain"]
PDB         = ROOT / cfg["structure"]["repaired_pdb"]
RES         = ROOT / "results"

# ---------------------------------------------------------------- load + M2
A   = pd.read_csv(RES / "listA_variants.csv")
mem = pd.read_csv(RES / "patch_membership.csv")
ala = pd.read_csv(RES / "ala_scan.csv")

# collapse List A to one row per residue
perpos = (A.groupby(["position", "wt"], as_index=False)
            .agg(n_variants=("ddg", "size"),
                 mean_ddg=("ddg", "mean"),
                 min_ddg=("ddg", "min")))
perpos["deleterious"] = perpos["min_ddg"] < DEL_BELOW

merged = (perpos
          .merge(mem, on="position", how="left")
          .merge(ala[["position", "ddG"]].rename(columns={"ddG": "ala_ddg"}),
                 on="position", how="left"))
merged["in_patch"] = merged["in_patch"].fillna(False)
merged = merged.sort_values("position").reset_index(drop=True)
merged.to_csv(RES / "merged.csv", index=False)

# ---------------------------------------------------------------- M3 Fisher
def two_by_two(df):
    a = int(((df.deleterious) & (df.in_patch)).sum())
    b = int(((df.deleterious) & (~df.in_patch)).sum())
    c = int(((~df.deleterious) & (df.in_patch)).sum())
    d = int(((~df.deleterious) & (~df.in_patch)).sum())
    return a, b, c, d

a, b, c, d = two_by_two(merged)
table = [[a, b], [c, d]]
or_two, p_two = fisher_exact(table, alternative="two-sided")
_,     p_grt  = fisher_exact(table, alternative="greater")

# ---------------------------------------------------------------- M4 Spearman
m4 = merged.dropna(subset=["ala_ddg"]).copy()
m4["abs_exp"] = m4["mean_ddg"].abs()
rho_all, p_all = spearmanr(m4["abs_exp"], m4["ala_ddg"])
inp = m4[m4.in_patch]
if len(inp) >= 3:
    rho_in, p_in = spearmanr(inp["abs_exp"], inp["ala_ddg"])
else:
    rho_in, p_in = float("nan"), float("nan")

# ---------------------------------------------------------------- M5 freesasa
# Extract chain A protein residues to a temp PDB, compute SASA, get relative SASA
class ChainAProt(Select):
    def accept_chain(self, ch): return ch.id == CHAIN
    def accept_residue(self, res): return res.id[0] == " "      # standard residues only
structure = PDBParser(QUIET=True).get_structure("p", str(PDB))
io = PDBIO(); io.set_structure(structure)
tmp = RES / "_chainA_for_sasa.pdb"
io.save(str(tmp), ChainAProt())

MAXASA = {  # Tien et al. 2013 theoretical maximum ASA (A^2), 1-letter
 "A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,"I":197,
 "L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}

sas = freesasa.calc(freesasa.Structure(str(tmp)))
areas = sas.residueAreas()[CHAIN]
relsasa = {}
for resnum, ra in areas.items():
    try:
        p = int(resnum)
    except ValueError:
        continue
    wt = ala.loc[ala.position == p, "residue"]
    if len(wt):
        relsasa[p] = ra.total / MAXASA.get(wt.iloc[0], 999)

merged["rel_sasa"] = merged["position"].map(relsasa)
merged["buried"] = merged["rel_sasa"] < BURIED_REL
merged.to_csv(RES / "merged.csv", index=False)   # re-save with sasa columns

buried_df  = merged[merged.buried == True]
exposed_df = merged[merged.buried == False]

def fisher_block(df):
    a, b, c, d = two_by_two(df)
    if min(a + b, c + d, a + c, b + d) == 0:
        orr, p = float("nan"), float("nan")
    else:
        orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return (a, b, c, d), orr, p

(bt, bor, bp) = fisher_block(buried_df)
(et, eor, ep) = fisher_block(exposed_df)

# ---------------------------------------------------------------- print
print("=== M3: 2x2 (residue level) ===")
print(f"                in_patch   not_in_patch")
print(f"deleterious        {a:>3}        {b:>3}")
print(f"not_deleterious    {c:>3}        {d:>3}")
print(f"odds ratio = {or_two:.3g}   p(two-sided) = {p_two:.4g}   p(greater) = {p_grt:.4g}")
print(f"\n=== M4: Spearman |exp ddG| vs alanine-scan ddG ===")
print(f"all residues (n={len(m4)}): rho={rho_all:.3f}  p={p_all:.4g}")
print(f"in-patch only (n={len(inp)}): rho={rho_in:.3f}  p={p_in:.4g}")
print(f"\n=== M5: confounder (buried < {BURIED_REL} rel SASA) ===")
print(f"buried positions:  {len(buried_df)}   exposed: {len(exposed_df)}")
print(f"buried  2x2 (a,b,c,d)={bt}  OR={bor:.3g}  p(greater)={bp:.4g}")
print(f"exposed 2x2 (a,b,c,d)={et}  OR={eor:.3g}  p(greater)={ep:.4g}")

# cross-tab in_patch vs buried (shows the confound)
ct = pd.crosstab(merged.buried, merged.in_patch)
print("\nin_patch vs buried among test residues:")
print(ct.to_string())

# ---------------------------------------------------------------- M6 REPORT
def pct(x, n): return f"{100*x/n:.0f}%" if n else "n/a"
N = len(merged)
report = f"""# p53 hotspot-patch vs experimental stability - RESULTS

**Hypothesis tested:** residues that belong to a structural hotspot patch are more
likely to be residues whose mutation destabilises the protein.

Two lists were built completely independently and only joined at the end:
- **List A (experiment):** ThermoMutDB measured ddG, single-point mutations, region {cfg['protein']['region_start']}-{cfg['protein']['region_end']}.
- **List B (structure):** FoldX alanine-scan hotspots, grouped into 3D patches.

## Data summary
- Residues with experimental data (test universe): **{N}**
- Deleterious residues (>=1 mutation with ddG < {DEL_BELOW}): **{int(merged.deleterious.sum())}**
- Residues in the hotspot patch (List B membership): **{int(merged.in_patch.sum())}** of {N}

## M3 - Main test (Fisher exact, residue level)
|                | in_patch | not_in_patch |
|----------------|---------:|-------------:|
| deleterious    | {a} | {b} |
| not_deleterious| {c} | {d} |

- Odds ratio = **{or_two:.3g}**
- p-value (two-sided) = **{p_two:.4g}**
- p-value (one-sided, enrichment) = **{p_grt:.4g}**

Plain English: an odds ratio above 1 means deleterious residues are *more* common
inside the hotspot patch than outside. Here OR = {or_two:.2g}; the enrichment is
{'statistically significant' if p_grt < 0.05 else 'NOT statistically significant'}
at p < 0.05 (one-sided p = {p_grt:.3g}).

## M4 - Intensity test (Spearman)
Because all hotspots form a single patch (constant patch_score), intensity was
measured per residue as the alanine-scan ddG (predicted hotspot strength).

- All residues with experimental data (n={len(m4)}): rho = **{rho_all:.3f}**, p = **{p_all:.4g}**
- In-patch residues only (n={len(inp)}): rho = **{rho_in:.3f}**, p = **{p_in:.4g}**

Plain English: a positive rho means residues predicted to be stronger hotspots
also tend to have larger measured destabilisation. The all-residue correlation is
rho = {rho_all:.2f} ({'significant' if p_all < 0.05 else 'not significant'} at p<0.05).

## M5 - Confounder check (burial)
Buried residues tend to be both hotspots and destabilisation-sensitive, so we
repeat the Fisher test separately for buried and exposed residues
(buried = relative SASA < {BURIED_REL}).

- Buried positions: {len(buried_df)}  |  Exposed positions: {len(exposed_df)}
- Buried stratum:  2x2 (a,b,c,d) = {bt}, OR = {bor:.3g}, p(greater) = {bp:.4g}
- Exposed stratum: 2x2 (a,b,c,d) = {et}, OR = {eor:.3g}, p(greater) = {ep:.4g}

Cross-tab of in_patch vs buried (shows how linked they are):

{ct.to_string()}

## Verdict
"""
marginal_support = (p_grt < 0.05 and or_two > 1)
# is the patch effect confounded by burial? (patch ~ buried core, and the
# association weakens / cannot be shown within burial strata)
exposed_inpatch = int(((merged.buried == False) & (merged.in_patch)).sum())
confounded = (exposed_inpatch == 0) and not (bp < 0.05)

if marginal_support and confounded:
    report += (
        f"**Qualified / partial support.** At face value the hypothesis is supported "
        f"(main test OR = {or_two:.2g}, one-sided p = {p_grt:.3g}; intensity rho = {rho_all:.2f}, "
        f"p = {p_all:.3g}). HOWEVER, the confounder check shows the hotspot patch is almost "
        f"identical to the buried core: all {int(merged[merged.in_patch].shape[0])} in-patch "
        f"residues are buried and no exposed residue is in the patch. Within the buried residues "
        f"the enrichment is weaker and not significant (OR = {bor:.2g}, p = {bp:.2g}), and the "
        f"exposed stratum cannot be tested (no in-patch residues there). "
        f"**Conclusion: the data are consistent with the hypothesis, but the effect cannot be "
        f"separated from simple burial - 'hotspot patch' and 'buried core' are the same residues "
        f"in this dataset.**\n")
elif marginal_support:
    report += (f"The data **SUPPORT** the hypothesis (main test OR = {or_two:.2g}, one-sided "
               f"p = {p_grt:.3g}), and the effect persists in the burial check (buried-stratum "
               f"OR = {bor:.2g}, p = {bp:.2g}).\n")
else:
    report += (f"The data **do NOT clearly support** the hypothesis "
               f"(main test OR = {or_two:.2g}, one-sided p = {p_grt:.3g}).\n")

report += f"""
## Cutoffs used (from config.yaml)
- Region of interest: {cfg['protein']['region_start']}-{cfg['protein']['region_end']}
- Deleterious (destabilising): experimental ddG < {DEL_BELOW} kcal/mol (ThermoMutDB sign convention)
- Stabilising: experimental ddG > {STAB_ABOVE} kcal/mol
- Alanine-scan hotspot: ddG >= {HOT} kcal/mol
- Patch distance: closest heavy atoms within {DIST} Angstrom
- Buried: relative SASA < {BURIED_REL}
- Structure: chain {CHAIN} of {cfg['structure']['repaired_pdb']} (zinc present)
"""
(RES / "REPORT.md").write_text(report, encoding="utf-8")
tmp.unlink(missing_ok=True)
print(f"\nSaved -> results/merged.csv and results/REPORT.md")
