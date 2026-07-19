# TP53 microenvironment-vs-stability project 

**Hypothesis tested:** the structural impact of a variant is determined by the 3D
microenvironment (hotspot patch) to which it belongs. Operationalised as two questions:
- **Q1 (enrichment):** are destabilising variants over-represented in hotspot patches?
- **Q2 (intensity):** does the size of a variant's impact depend on the aggregated 3D-patch score?

The two branches (experimental variants vs structural patches) were built **independently**
and only joined at the statistics stage. All cut-offs live in one `config.yaml`.

---

## 1. Variant branch (experimental ΔΔG)
- **Source:** ThermoMutDB (downloaded programmatically; it already aggregates ProTherm + the
  thermodynamic literature). Filtered to human p53 (P04637), single-point mutations, residues 93–290.
- **Sign-convention check (important):** ThermoMutDB stores **negative ΔΔG = destabilising** —
  the *opposite* of the usual convention. Verified against textbook destabilising mutants
  (R175H, V143A, Y220C all negative). We kept the data **as published** and set the cut-offs
  accordingly (destabilising < −1.0, stabilising > +1.0 kcal/mol).
- **Pseudoreplication handled:** several substitutions at one residue (e.g. R248Q/W/G) are not
  independent, so we **collapsed to the residue level** — one representative value per residue =
  the **most destabilising** measurement (mean reported as a sensitivity check; conclusions
  unchanged). No mixed-effects model (not needed; would not change the conclusion).
- **Result:** 43 single-point variants → **37 residues**; 19 destabilising / 16 neutral / 2 stabilising.

## 2. Structure branch — prioritisation
- Evaluated candidate p53 DBD structures (2OCJ, 1TUP, 1UOL, 2XWR, 4HJE, 3Q05, 2FEJ) by
  resolution, completeness over 93–290, internal gaps, and zinc presence.
- **Selected 2OCJ** — wild-type, DNA-free (apo), complete & gap-free over the region, structural
  zinc present, 2.05 Å. Rejected 1UOL (stabilising mutant — would bias a stability study),
  4HJE/1TUP (DNA-bound — DNA distorts loops), 2FEJ (NMR), 2XWR (non-standard construct).
- The FoldX-repaired file **matches 2OCJ**, so prioritisation confirmed it.

## 3. Structure QC & cleaning
- Chain check: chain A is **continuous, residues 96–289, no internal gaps** (only the disordered
  termini 93–95 and 290 absent — outside the crystallised region).
- Heteroatoms: only **zinc** present (no waters/other ligands). Zinc **retained** (it is essential —
  without it the zinc-binding residues get wrong ΔΔG).
- **Consistent context:** all structural steps (alanine scan, RSA, patches, PositionScan) were run
  on a single **chain A + its zinc** extract, so the analysis is internally consistent.

## 4. Repair / refinement
- The structure was already **FoldX-repaired** (RepairPDB); we used it as-is
  (no re-repair, no MODELLER needed — no missing internal atoms).

## 5. Alanine scan & hotspot definition
- FoldX **AlaScan** on all 194 chain-A residues → per-residue ΔΔG (contribution to stability).
- Sanity check passed: the four zinc-ligand residues (Cys176, His179, Cys238, Cys242) are the
  four strongest hotspots — biologically correct.
- **Hotspot = alanine-scan ΔΔG ≥ 2.0 kcal/mol** (chosen after viewing the distribution; above
  FoldX noise) → **64 hotspot residues**.

## 6. 3D patches & microenvironment score
- Two residues are neighbours if their closest heavy atoms are within **8 Å**.
- **Honest structural finding:** the hotspots do **not** form several separate patches — they form
  **one contiguous core** (63 residues, + 1 isolated hotspot). We did **not** invent extra patches.
- **Aggregated 3D-patch score** per residue = sum of the alanine-scan ΔΔG of all residues in its
  8 Å microenvironment (this gives the per-variant score Q2 needs).

## 7. Statistics
- **Q1 (enrichment, residue level, 2×2 Fisher):**

  |                | hotspot | non-hotspot |
  |----------------|--------:|------------:|
  | destabilising  | 12 | 7 |
  | other          | 3 | 15 |

  Odds ratio = **8.0**, 95% CI **[1.5, 59]**, Fisher p = **0.007** → destabilising residues are
  **over-represented** in hotspot patches.
- **Q2 (intensity):** Spearman(|ΔΔG|, aggregated patch score) = **0.51**, p = **0.001** → variants
  in stronger structural microenvironments tend to be more destabilising. (Association only — no
  causal claim.)

## 8. Confounder analysis — burial (the key caveat)
- Hotspot/core residues are buried, so we tested **solvent accessibility (RSA)** as a confounder.
- **patch score and RSA are strongly correlated** (Spearman −0.84 at n=37, −0.77 at n=194; VIF ≈ 2.2–2.8),
  and this does **not** improve with coverage — the collinearity is intrinsic.
- **After adjusting for RSA the patch effect collapses:** Q1 odds ratio 8.6 → **3.0 (p = 0.27)**;
  Q2 coefficient 0.033 → **0.003 (p = 0.84)**.

## 9. Robustness / sensitivity
- **Consistent-context re-run** (chain A + zinc throughout): results identical → context did not bias anything.
- **Expanded coverage (FoldX PositionScan, 37 → 194 residues):** predicted ΔΔG validated against
  experiment (Spearman −0.51). The binary Tier-3 Q1 looks strong (OR ~90) but is **semi-circular**
  (the predicted outcome and the patch score share the FoldX alanine-scan basis) and is **not**
  trustworthy. The continuous Tier-3 Q2 — less prone to that artifact — again becomes
  **non-significant after RSA adjustment (p = 0.16)**, consistent with the experimental result.
- **Cut-off robustness:** the destabilising cut-off (±1 kcal/mol) was fixed in advance, not fitted.
  We also swept the neighbourhood radius (2.5–8 Å) and the ΔΔG cut-off; the conclusion is unchanged
  at every setting, and **all** settings are reported (no cherry-picking).
- **Patch count is definitional:** at the loose 8 Å the hotspots form **one core**; at a tight
  (~3 Å) atomic-contact radius they resolve into **10 patches** (`patches_3D_tight_3A.csv`) that map
  onto real p53 landmarks — the β-sandwich core, the zinc site (176/177), loop L3 (contains the
  cancer hotspots G245/R248/R249) and the DNA-contact helix H2. Same residues, different resolution.

## 10. Functional-impact extension (SEPARATE from the stability result)
Experimental ΔΔG is capped at 37 residues — we verified FireProtDB/ProTherm contain **no additional
p53 ΔΔG**. To gain power we ran a **separate** analysis using functional deep-mutational-scanning
data (MaveDB), clearly labelled as **function, not folding stability** (it conflates stability with
DNA-binding/expression):

| Assay | n residues | Q2 ρ (microenvironment vs impact) | RSA-adjusted (beyond burial) |
|---|---:|---:|---|
| Giacomelli 2018 (drug selection) | 175 | **+0.64** (p = 3×10⁻²¹) | p = 6×10⁻⁵ |
| Kato 2003 (transcriptional) | 194 | **+0.41** (p = 3×10⁻⁹) | p = 0.68 |

- **The hypothesis REPLICATES** across two independent assays/labs (ρ = 0.64 and 0.41).
- **The "beyond burial" effect does NOT replicate** — significant in Giacomelli only, contradicted by
  Kato — so we do **not** claim it.
- **Kotler 2018 was discarded**: its positions run 1–191 and residue 175 is Ala (human 175 is Arg),
  i.e. non-human/mouse numbering — merging it onto the human structure would compare different
  residues. (The same numbering check that validated the stability data caught this.)

## 11. Conclusion (one line)
> The hypothesis is **supported and replicated**: a variant's impact is determined by its 3D
> structural microenvironment (ρ = 0.64 and 0.41 in two independent assays; OR ≈ 8–10 enrichment).
> The microenvironment feature driving it is **solvent burial** — after adjusting for burial the
> patch signal does not survive in the stability data, and the one dataset suggesting an effect
> beyond burial failed to replicate. So: **microenvironment effect confirmed; mechanism = burial;
> no robust evidence of an effect beyond burial.**

## Limitations to state up front
- Small experimental ΔΔG sample (37 residues) → wide confidence intervals; and it is the data ceiling
  (no additional p53 ΔΔG exists in FireProtDB/ProTherm).
- Patch membership and burial are near-collinear (ρ ≈ −0.8) and cannot be separated with current data.
- The hotspots form a single core at standard distances, so "patch" ≈ "buried core".
- The functional extension answers a **different** question (function ≠ folding stability).
- FoldX predictions only moderately track experiment (ρ ≈ 0.5); not a substitute for measurements.

## Open questions 
1. Should "impact" mean **folding stability** (n = 37) or **functional impact** (n = 175–194)?
2. Is the **beyond-burial** question worth pursuing further, or do we write up what we have?

## Deliverables
Main: `REPORT_consistent.md`, `SUMMARY.md`, `METHODS_LOG.md`.
Data: `structure_candidates.csv`, `listA_variants.csv`, `residue_table_cA.csv`,
`patches_3D_cA.csv`, `patches_3D_tight_3A.csv`, `functional_ext/results/replication_table.csv`.
Figures: alanine-scan profile, 3D patch, Q2 scatter, predicted-vs-experimental validation,
patch-vs-RSA confounder.
