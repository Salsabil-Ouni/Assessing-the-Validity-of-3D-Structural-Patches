# FUNCTIONAL-IMPACT EXTENSION (separate from the stability result)

**Outcome:** Giacomelli 2018 combined loss-of-function score (MaveDB urn:00000068-0-1);
higher = more loss of function. **This is FUNCTION, not folding stability** - it mixes
stability with DNA-binding/expression, so it is reported separately and NOT merged with
the 37-residue ΔΔG result. Its value is coverage/power: **175 residues** (96-270).

**Structural predictors** (patch_score, in_patch, RSA) are from the stability pipeline.

## Q1 - enrichment (functionally-damaging vs in hotspot patch)
Full (n=175) and buried-only (n=56) - see printed table; OR + 95% CI + Fisher p.

## Q2 - intensity (functional impact vs microenvironment score)
- Full: Spearman rho, p (printed)
- Buried-only (the powered 'beyond burial' test): Spearman rho, p (printed)
- RSA-adjusted OLS: patch coefficient before/after adding RSA (printed)

## How to read it
If the patch effect on FUNCTIONAL impact survives RSA adjustment / holds within buried
residues (now with ~56 buried residues, not 18), that is evidence the structural
microenvironment predicts functional impact beyond burial. Interpret cautiously: functional
score is a confounded proxy, and this answers a DIFFERENT question than the stability hypothesis.
