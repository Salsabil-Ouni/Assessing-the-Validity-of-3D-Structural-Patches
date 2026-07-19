# TP53 — do 3D structural microenvironments determine variant impact?

Test of the hypothesis: **the structural impact of a missense variant is determined by the 3D
microenvironment (hotspot patch) to which it belongs**, using the p53 DNA-binding domain.

Two operational questions:
- **Q1 (enrichment):** are destabilising variants over-represented in hotspot patches?
- **Q2 (intensity):** does variant impact scale with the aggregated 3D-patch score of its microenvironment?

The experimental (variant) and structural (patch) branches are built **independently** and joined
only at the statistics stage. Every cut-off lives in `config.yaml`.

---

## TL;DR result

- **Hypothesis supported, and it replicates.** Destabilising variants are ~8× enriched in hotspot
  patches (Fisher OR 8.0, 95% CI [1.5, 59], p = 0.007), and impact rises with the microenvironment
  score (Spearman ρ = 0.51, p = 0.001). In an independent functional-impact extension the effect
  replicates across two assays (ρ = 0.64 and ρ = 0.41, both p < 10⁻⁸).
- **Mechanism = solvent burial.** The hotspot patch coincides with the buried core (patch score vs
  RSA ρ ≈ −0.8). After adjusting for burial the patch signal does not survive in the stability data;
  a hint of an effect beyond burial in one functional dataset **failed to replicate**.
- **Net:** microenvironment effect confirmed; the driving feature is burial; **no robust evidence of
  an effect beyond burial** with current data.

---

## Data sources

| Branch | Source | Use |
|---|---|---|
| Structure | **PDB 2OCJ** (WT human p53 core domain, DNA-free, 2.05 Å, zinc present) | coordinates for alanine scan, patches, burial |
| Experimental ΔΔG | **ThermoMutDB** (aggregates ProTherm + literature) | 37 residues with single-point ΔΔG in 93–290 |
| Functional (extension) | **MaveDB** deep mutational scans: Giacomelli 2018, Kato 2003 | 175 / 194 residues, function (kept separate) |

**Sign convention:** ThermoMutDB stores **negative ΔΔG = destabilising** (opposite to the usual
convention); verified against known destabilisers (R175H, V143A, Y220C) and encoded in `config.yaml`
(destabilising < −1, stabilising > +1 kcal/mol). Data used as published.

**Data ceiling:** FireProtDB and ProTherm contain **no additional p53 ΔΔG** beyond ThermoMutDB, so 37
residues is the experimental limit.

---

## Pipeline (methods)

1. **Build List A** — ThermoMutDB → single-point mutations in 93–290, duplicates averaged, labelled
   destabilising / neutral / stabilising. Collapsed to **residue level** (representative = the most
   destabilising measurement) to avoid pseudo-replication. → 37 residues.
2. **Structure prioritisation** — ranked candidate PDBs (2OCJ, 1TUP, 1UOL, 2XWR, 4HJE, 2FEJ, 3Q05) by
   resolution, completeness over 93–290, gaps, and zinc; selected **2OCJ** (WT, apo). Rejected the
   stabilising mutant 1UOL and DNA-bound 1TUP/4HJE.
3. **QC** — chain A is continuous (96–289, no internal gaps); only heteroatom is the structural
   **zinc**, which is retained. All structural steps use one consistent context: **chain A + its zinc**.
4. **Alanine scan (FoldX)** — per-residue ΔΔG (contribution to stability); the four zinc-ligand
   residues (Cys176/His179/Cys238/Cys242) are the top hotspots (sanity check passed).
5. **Hotspots + patches** — hotspot = alanine-scan ΔΔG ≥ 2.0 (64 residues); two residues linked if
   closest heavy atoms ≤ 8 Å; connected components = patches. Result: **one contiguous core** (63) +
   a singleton. Each residue gets an aggregated microenvironment score (Σ neighbour ΔΔG within 8 Å).
6. **Numbering check** — List-A residue letters vs structure: **37/37 match** (safe to merge).
7. **Statistics** — Q1 Fisher exact (OR + 95% CI), Q2 Spearman, at residue level.
8. **Confounder (burial)** — relative solvent accessibility (freesasa); correlation with patch score,
   VIF, and RSA-adjusted logistic/linear models; buried-only stratified test.
9. **Robustness** — consistent-context re-run (identical), distance sweep (2.5–8 Å), ΔΔG cut-off
   sweep, and a FoldX PositionScan extension to all 194 residues (predicted ΔΔG).
10. **Functional-impact extension** — MaveDB deep mutational scans, auto-oriented so higher = more
    damaging, run through the same Q1/Q2; replicated across assays.

---

## Results

### Q1 — enrichment (residue level)

|                | hotspot | non-hotspot |
|----------------|--------:|------------:|
| destabilising  | 12 | 7 |
| other          | 3 | 15 |

Fisher **OR = 8.0**, 95% CI **[1.5, 59]**, p = **0.007**.

### Q2 — intensity
Spearman(|ΔΔG|, aggregated patch score) = **0.51**, p = **0.001**.

### Confounder — burial
- corr(patch score, RSA): **−0.84** (n = 37), **−0.77** (n = 194); VIF ≈ 2.2–2.8 (intrinsic, does not
  improve with coverage).
- RSA-adjusted: Q1 OR 8.6 → **3.0 (p = 0.27)**; Q2 coefficient 0.033 → **0.003 (p = 0.84)** → the
  patch effect does not survive burial adjustment.

### Robustness
- **Distance / cut-off sweeps:** conclusion unchanged at every setting; all reported (no cherry-pick).
- **Patch count is definitional:** at 8 Å the hotspots form one core; at a tight ~3 Å atomic-contact
  radius they resolve into **10 patches** (`results/patches_3D_tight_3A.csv`) mapping onto real p53
  landmarks — β-sandwich core, zinc site (176/177), loop L3 (contains cancer hotspots G245/R248/R249),
  DNA-contact helix H2.
- **PositionScan extension (n = 194):** predicted ΔΔG validates against experiment (ρ = −0.51); the
  continuous test again loses significance after RSA adjustment (p = 0.16). (The binary version is
  semi-circular and is not used.)

### Functional-impact extension (separate from stability)

| Assay | n residues | Q2 ρ | RSA-adjusted (beyond burial) |
|---|---:|---:|---|
| Giacomelli 2018 (drug selection) | 175 | **+0.64** (p = 3×10⁻²¹) | p = 6×10⁻⁵ |
| Kato 2003 (transcriptional) | 194 | **+0.41** (p = 3×10⁻⁹) | p = 0.68 |

The hypothesis **replicates** in both. The "beyond burial" effect appears only in Giacomelli and is
contradicted by Kato, so it is **not claimed**. Kotler 2018 was **discarded** — its positions run
1–191 and residue 175 is Ala (human 175 is Arg), i.e. non-human numbering (caught by the same
numbering check used in step 6).

---

## Conclusion

> A variant's structural impact is determined by its 3D microenvironment — **supported and replicated**
> across independent assays. The microenvironment feature driving it is **solvent burial / core
> packing**; after adjusting for burial the patch signal does not persist, and the single dataset
> hinting at an effect beyond burial failed to replicate. **Microenvironment effect confirmed;
> mechanism = burial; no robust effect beyond burial.**

## Limitations
- Experimental ΔΔG covers only 37 residues (the data ceiling) → wide confidence intervals.
- Patch membership and burial are near-collinear (ρ ≈ −0.8) and cannot be fully separated here.
- Hotspots form a single core at standard distances, so "patch" ≈ "buried core".
- The functional extension answers a **different** question (function ≠ folding stability).
- FoldX predictions only moderately track experiment (ρ ≈ 0.5).

## Open questions
1. Should "impact" mean **folding stability** (n = 37) or **functional impact** (n = 175–194)?
2. Is the **beyond-burial** question worth pursuing further, or write up as-is?

---

## Repository structure

```
config.yaml                 all cut-offs (region, ΔΔG ±1, hotspot ≥2, patch 8 Å, buried RSA<0.2)
structure/                  2OCJ-derived PDBs (tp53_repaired.pdb, tp53_chainA_zinc.pdb)
data/                       downloaded databases (ThermoMutDB, candidate PDBs)
scripts/                    pipeline scripts (see map below)
results/                    reports, tables, figures/
functional_ext/             functional-impact extension (data, scripts, results) — SEPARATE
```

### Key outputs (`results/`)
- `SUPERVISOR_SUMMARY.md` (overview) · `REPORT_consistent.md` (full stability) · `METHODS_LOG.md`
- `listA_variants.csv`, `residue_table_cA.csv`, `patches_3D_cA.csv`, `patches_3D_tight_3A.csv`,
  `structure_candidates.csv`
- `figures/` — alanine-scan profile, 3D patch, Q2 scatter, predicted-vs-experimental validation,
  patch-vs-RSA confounder
- `functional_ext/results/replication_table.csv`

### Script → step map

| Step | Script |
|---|---|
| Build List A | `scripts/build_listA.py` |
| Structure prioritisation | `scripts/structure_prioritization.py` |
| Structure QC | `scripts/inspect_structure.py` |
| Alanine scan (FoldX/WSL) | `scripts/run_alascan_chainA.sh` → `scripts/build_ala_scan.py` |
| Hotspots + patches | `scripts/build_patches.py` |
| Numbering check | `scripts/check_numbering.py` |
| Main stats (residue-level) | `scripts/prof_pipeline_strict.py` |
| Confounder / burial | `scripts/extended_analysis.py`, `scripts/burial_model.py` |
| Consistent context | `scripts/consistent_context.py` |
| Robustness sweeps | `scripts/buried_only.py`, `scripts/distance_sweep.py`, `scripts/sensitivity_cutoff.py` |
| PositionScan extension | `scripts/run_posscan_full.sh`, `scripts/predicted_tier.py` |
| Figures | `scripts/make_figures_cA.py`, `scripts/make_patch3d.py` |
| Functional extension + replication | `functional_ext/scripts/functional_analysis.py`, `replication.py` |

---

## Reproducing

- **Python** (venv): `pandas numpy scipy statsmodels biopython freesasa networkx requests pyyaml matplotlib`.
- **FoldX** (academic licence) for the alanine scan and PositionScan; run here via WSL (Linux build).
- Scripts read settings from `config.yaml` and reference local paths (FoldX location, WSL mounts), so
  they document the exact workflow rather than being a turnkey one-command re-run.
