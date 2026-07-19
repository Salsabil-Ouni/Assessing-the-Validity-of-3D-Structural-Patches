# Exactly what was done — per pipeline step (reproducible log)

## 0. Environment
- Created folders `data/ structure/ results/ scripts/` in `D:\Desktop\test 2`.
- Created venv: `python -m venv .venv` (Anaconda Python 3.10.9).
- Installed: `pandas numpy scipy biopython freesasa networkx requests pyyaml` (later `statsmodels matplotlib`).
- Wrote `config.yaml` with all cut-offs.
- Copied `tp53_Repair.pdb` → `structure/tp53_repaired.pdb` (original kept as backup).

## 1. Variant branch — get experimental ΔΔG
- Downloaded ThermoMutDB full database:
  `curl -o data/thermomutdb.json https://biosig.lab.uq.edu.au/thermomutdb/static/data/thermomutdb.json` (17 MB, 13,337 records).
- Filtered records to `protein == "Cellular tumor antigen p53"` → 133 records.

## 2. Variant branch — build List A  (`scripts/build_listA.py`)
- Kept only single-point codes matching regex `^[A-Z]\d+[A-Z]$` with a numeric ΔΔG and position in 93–290.
- Collapsed duplicate measurements of the **same** mutation by **mean**.
- Labelled by config cut-offs.
- Output `results/listA_variants.csv` → **43 variants** (20 deleterious / 21 neutral / 2 stabilising at variant level).

## 3. Structure inspection  (`scripts/inspect_structure.py`, Biopython)
- Found **4 chains A–D**, residues **96–289**, **4 zinc ions** (`ZN` 501–504). Wrote `results/structure_info.txt`. Chose **chain A**.

## 4. FoldX setup
- Located FoldX at `D:\data D\Downloads D\foldx_Linux\foldx_20261231` (Linux ELF binary). It runs via **WSL Ubuntu-24.04**.
- The other copy (`foldx_20251231`) was **licence-expired**; used the valid `…20261231`.
- Recorded WSL paths in `config.yaml` (`foldx.binary_wsl`, `foldx.workdir_wsl`).
- Read FoldX **AlaScan** manual (web) to confirm `--command=AlaScan` syntax and output format.

## 5. Alanine scan (initial, tetramer)  (`scripts/run_alascan.sh` via WSL)
- Ran `foldx --command=AlaScan --pdb=tp53_repaired.pdb` in a space-free `/tmp` dir (path-with-space workaround); copied `tp53_repaired_AS.fxout` (776 lines = 4×194) to `results/foldx_raw/`.

## 6. Tidy alanine scan  (`scripts/build_ala_scan.py`)
- Took the **first block** (= chain A) up to the residue-number reset.
- Fixed FoldX histidine codes (`H1S/H2S` → H) — initially 7 His were dropped; after fix → **194 residues**.
- Output `results/ala_scan.csv`. Sanity check: top hotspots = Cys176/His179/Cys238/Cys242 (the zinc ligands).

## 7. Hotspot threshold + patches  (`scripts/build_patches.py`, networkx)
- Showed ΔΔG distribution; **hotspot cut-off chosen = 2.0** (interactive) → 64 hotspots; wrote to `config.yaml`.
- Edge if closest heavy atoms ≤ **8 Å**; connected components = patches.
- Result: **one cluster of 63 + one singleton (226)**. Output `results/listB_patches.csv`, `patch_membership.csv`.

## 8. Numbering check (M1)  (`scripts/check_numbering.py`)
- Compared List-A wild-type letter vs structure residue for all **37** positions → **37 match, 0 mismatch** (R175=R, R248=R, etc.). Confirmed same numbering.

## 9. First merge + stats  (`scripts/merge_and_test.py`)
- Collapsed to residue level, Fisher (Q1), Spearman (Q2), freesasa for burial (M5), wrote `results/REPORT.md`.

## 10. Prof-aligned re-runs
- `scripts/prof_pipeline.py` — variant-level microenvironment score (per-variant 8 Å neighbour sum of alanine ΔΔG); Q1 (Fisher), Q2 (Spearman).
- `scripts/prof_pipeline_strict.py` — **residue-level**: representative ΔΔG = **most destabilising (min signed)**, mean as sensitivity; Q1 Fisher with **odds ratio + 95% CI** (`scipy.stats.contingency.odds_ratio`); logistic/OLS (`statsmodels`). Wrote `REPORT_prof.md`, `patches_3D.csv`, `variant_microenvironment.csv`.

## 11. Confounder / coverage  (`scripts/extended_analysis.py`)
- Computed `corr(patch_score, RSA)` (Spearman + Pearson) and **VIF** on n=37 and n=194; RSA-adjusted logistic (Q1) and OLS (Q2). Wrote `REPORT_extended.md`, `residue_table_full.csv`.

## 12. Structure prioritisation  (`scripts/structure_prioritization.py`)
- Downloaded candidates from `https://files.rcsb.org/download/<ID>.pdb` (2OCJ, 1TUP, 1UOL, 2XWR, 4HJE, 2FEJ, 3Q05).
- Parsed method, resolution (REMARK 2), protein chains, best-chain coverage of 93–290, internal gaps, zinc count, title → `results/structure_candidates.csv`.
- **Selected 2OCJ** (WT, apo, complete, gap-free, zinc, 2.05 Å). Confirmed the provided repaired file = FoldX-repaired 2OCJ.

## 13. Consistent context — chain A + zinc
- Extracted chain A + the **nearest zinc** (2.06 Å) → `structure/tp53_chainA_zinc.pdb` (Biopython select).
- Re-ran AlaScan on it (`scripts/run_alascan_chainA.sh`) → `results/foldx_raw/chainA_AS.fxout` (194 lines).
- `scripts/consistent_context.py`: re-parsed alanine scan, rebuilt patches (8 Å), RSA via **freesasa** (chain-A protein, Tien-2013 max-ASA), Q1/Q2 + RSA-adjusted + confounding; wrote `ala_scan_cA.csv`, `patches_3D_cA.csv`, `residue_table_cA.csv`, `residue_table_full_cA.csv`, `REPORT_consistent.md`. Results identical to tetramer.

## 14. PositionScan extension (Tier 3)
- Read FoldX **PositionScan** manual (`--positions=LC43a`, `a`=all 20; output `PS_*_scanning_output`).
- Built positions list `scripts/ps_positions.txt` (194 × `<wt>A<pos>a`).
- Ran `scripts/run_posscan_full.sh` on `tp53_chainA_zinc.pdb` in **WSL background** → `results/foldx_raw/positionscan_scanning_output.txt` (**4074 lines**, 213 min).
- Tier-3 analysis (in `consistent_context.py`): per-residue predicted ΔΔG = **mean over 19 non-self substitutions**; validated vs experiment (Spearman −0.51); Q1/Q2 with RSA adjustment; flagged the binary-Q1 semi-circularity.

## 15. Figures  (`scripts/make_figures_cA.py`, `scripts/make_patch3d.py`)
- `ala_scan_profile.png`, `patch_3D.png` (4-colour: destabilising in/out of patch + body cloud, 2 views),
  `Q2_scatter.png`, `validation_scatter.png` (pred vs exp), `confounding_scatter.png` (patch vs RSA),
  `patch_view.pml` (PyMOL script for chain A + zinc).

## Key results produced
- Q1 (residue level): table [[12,7],[3,15]], Fisher **OR 8.0, 95% CI [1.5, 59], p 0.007**.
- Q2: Spearman **0.51, p 0.001**.
- Confounder: corr(patch, RSA) **−0.84 (n=37) / −0.77 (n=194)**, VIF ≈ 2.2–2.8.
- RSA-adjusted: Q1 OR **8.6 → 3.0 (p 0.27)**; Q2 coef **0.033 → 0.003 (p 0.84)**.
- Tier 3 (n=194): validation **−0.51**; binary Q1 circular (discarded); continuous Q2 RSA-adjusted **p 0.16**.
- Conclusion: **no robust microenvironment effect beyond burial**.
