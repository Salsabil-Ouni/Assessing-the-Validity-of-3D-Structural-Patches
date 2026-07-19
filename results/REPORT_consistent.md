# TP53 microenvironment vs burial - CONSISTENT-CONTEXT analysis (chain A + zinc)

**Structure:** 2OCJ (WT human p53 DBD, DNA-free, 2.05 A, complete over 93-290, zinc present)
- selected by the prioritisation step (see results/structure_candidates.csv); the provided
  FoldX-repaired file matches 2OCJ. Chain A + its zinc used for ALL steps (alanine scan,
  RSA, patch construction, PositionScan) for a single consistent context.

**Aggregation:** residue level; representative = most destabilising measured ddG
(mean reported as sensitivity). No mixed-effects model.

## Hotspots / patches (chain-A context)
hotspots (ala-scan ddG >= 2.0): 64; patches [63, 1];
residues in a patch (size>=2): 63.

## Confounding: patch_score vs RSA (outcome-independent)
| set | n | Spearman | Pearson | VIF |
|-----|--:|---------:|--------:|----:|
| experimental | 37 | -0.841 | -0.798 | 2.75 |
| full structural | 194 | -0.768 | -0.735 | 2.18 |

## Tier 1 - experimental (n=37)
- Q1 Fisher: table (a,b,c,d) = (12, 7, 3, 15), OR = 8.02, 95% CI [1.51, 58.95], p = 0.006941
- Q1 logistic patch OR: unadjusted 8.57 (p=0.00663);
  RSA-adjusted 3.05 (p=0.273)
- Q2 Spearman(|rep ddG|, patch_score): rho = 0.51, p = 0.00135
- Q2 OLS patch coef: unadjusted 0.0327 (p=0.00101);
  RSA-adjusted 0.0029 (p=0.837)

## Tier 3 - FoldX-predicted (PositionScan, coverage 37 -> 194)
- Validation Spearman(predicted, experimental ddG) = -0.51 (p=0.00124), negative as expected.
- Q1 Fisher OR=92.51 CI[15.00,3826.00] p=1.09e-17; logistic patch OR unadj=94.19 (p=9e-06), RSA-adjusted=51.66 (p=0.000139).
- Q2 Spearman rho=0.56 (p=2.85e-17); OLS patch coef unadj=0.0494 (p=4.19e-10), RSA-adjusted=0.0147 (p=0.164).

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
