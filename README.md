# 3D structural patches in TP53: pipeline and analyses

Code for testing whether the structural impact of a TP53 missense variant is determined by the
3D microenvironment (hotspot patch) it belongs to, using the p53 DNA-binding domain
(residues 93-290).

- **Q1 (enrichment):** are destabilising variants over-represented in hotspot patches?
- **Q2 (intensity):** does variant impact scale with the aggregated 3D-patch score of its microenvironment?

Experimental (variant) and structural (patch) branches are built independently and joined only at
the statistics stage. All cut-offs live in `config.yaml`.

## Headline results

| Test | Result |
|---|---|
| Q1, destabilising variants in hotspot patches | Fisher OR 8.0, 95% CI [1.5, 59], p = 0.007 |
| Q2, impact vs patch score | Spearman rho = 0.51, p = 0.001 |
| Functional-assay replication | rho = 0.64 and 0.41 across two assays, both p < 1e-8 |
| After adjusting for solvent burial | patch signal does not survive in the stability data |

Patch score tracks solvent burial (patch score vs RSA, rho about -0.8). Result tables are in
`results/`, `functional_ext/results/` and the per-analysis folders below.

## Pipeline

Two roads that meet at the merge step. Each script's docstring states its step.

**Road A: experimental variants**
- `scripts/build_listA.py`: ThermoMutDB ddG values to labelled variants (deleterious, neutral, stabilising)

**Road B: structure and patches**
- `scripts/inspect_structure.py`: structure checks (chain, zinc ion)
- `scripts/run_alascan*.sh`, `scripts/run_posscan_full.sh`: FoldX alanine scan and PositionScan (via WSL)
- `scripts/build_ala_scan.py`: tidy alanine-scan output
- `scripts/build_patches.py`: group hotspot residues into 3D patches (8 A neighbour graph)

**Merge and test**
- `scripts/check_numbering.py`: confirm experiment and structure use the same residue numbering
- `scripts/merge_and_test.py`: Q1 and Q2 statistics

**Sensitivity and confounders** (`scripts/`)
- `sensitivity_cutoff.py`, `distance_sweep.py`, `hotspot_threshold_sweep.py`, `representative_sensitivity.py`
- `burial_model.py`, `buried_only.py`, `own_value_confound.py`, `consistent_context.py`
- `extended_analysis.py`, `predicted_tier.py`, `mechanism_stratification.py`

**Second-structure validation** (2OCJ vs 2XWR)
- `validate_second_structure.py`, `second_structure_validation.py`, `patch_overlap_2ocj_2xwr.py`, `burial_confound_2xwr.py`

**DNA-interface test**
- `dna_proximity.py` (from 1TUP), `interface_beyond_burial.py`

**Figures**
- `make_figures*.py`, `make_patch3d.py`, `make_hotspot_threshold_figure.py`, `make_forest_and_buried_figures.py`

## Additional analyses

| Folder | Purpose |
|---|---|
| `functional_ext/` | Replication on independent functional assays (Giacomelli, Kato, Kotler) |
| `core_robustness_checks/` | Checks A to H: density-normalised score, equivalence/MDE, zinc leverage, numbering offset, spatial permutation null, assembly RSA, Q2 sensitivity, hotspot composition |
| `alphamissense_test/` | AlphaMissense comparison and stability leave-one-out |
| `variant_ml/` | Variant-level models with ESM-2 zero-shot scores (LightGBM) |
| `structure_ai_test/` | ProteinMPNN substitution scores and tests beyond burial |
| `interface_patch_test/` | Does spatial aggregation add value for an interface property burial cannot see |

## Requirements

Python 3 with: numpy, pandas, scipy, statsmodels, scikit-learn, lightgbm, biopython, freesasa,
networkx, matplotlib, pyyaml, requests. The ESM-2 and ProteinMPNN steps also need torch and
transformers. FoldX is a separate binary (run through WSL; paths are set in `config.yaml`).

## Data not included

Large and third-party inputs are not committed: ThermoMutDB export, MaveDB functional datasets,
AlphaMissense substitution table, raw ProteinMPNN outputs, and the PDB structures other than the
chain A file in `structure/` (2OCJ, 2XWR, 1TUP from the RCSB PDB).

## How to reproduce

1. Edit `config.yaml` to point at your own FoldX binary (via WSL) and confirm the structure paths.
2. Road A: run `scripts/build_listA.py` to turn the raw ThermoMutDB export into the labelled
   37-residue experimental set.
3. Road B: run `scripts/inspect_structure.py`, then the FoldX alanine scan and PositionScan
   shell scripts, then `scripts/build_ala_scan.py` and `scripts/build_patches.py`.
4. Run `scripts/check_numbering.py` to confirm the two roads share a numbering scheme, then
   `scripts/merge_and_test.py` for the Q1/Q2 statistics reported in the thesis.
5. The sensitivity, confounder, second-structure, replication, and exploratory ML/AI scripts
   under `scripts/`, `functional_ext/`, `core_robustness_checks/`, `alphamissense_test/`,
   `variant_ml/`, `structure_ai_test/`, and `interface_patch_test/` can each be run independently
   once steps 2-4 have produced their inputs; each script's docstring states what it needs.
6. Figures are regenerated with the `make_figures*.py` / `make_patch3d.py` /
   `make_hotspot_threshold_figure.py` / `make_forest_and_buried_figures.py` scripts.

## Citation

If you use this code, please cite the archived release rather than the live repository:

> Ouni, S. (2026). *Assessing the Validity of 3D Structural Patches for the Classification of
> Point Mutations: A Case Study on the TP53 DNA-Binding Domain* [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.24255701

## License

Released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), matching
the archived Zenodo deposit.
