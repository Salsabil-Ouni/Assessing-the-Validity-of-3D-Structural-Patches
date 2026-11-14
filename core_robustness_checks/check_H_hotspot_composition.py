"""Check H: hotspot vs non-hotspot residue composition (Aim 2's promised comparison, never delivered).
Hydrophobicity (Kyte-Doolittle), volume, charge, and packing density (RSA), plus amino-acid composition,
compared between hotspot and non-hotspot residues across all 194 residues, and separately for in-patch vs
outside-the-patch (a stricter split: hotspot status alone doesn't require spatial clustering)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,
          P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
VOL = dict(A=88.6,R=173.4,N=114.1,D=111.1,C=108.5,Q=143.8,E=138.4,G=60.1,H=153.2,I=166.7,L=166.7,K=168.6,
           M=162.9,F=189.9,P=112.7,S=89.0,T=116.1,W=227.8,Y=193.6,V=140.0)
CHG = dict(K=1, R=1, D=-1, E=-1)  # magnitude compared as |charge|

full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")
ala = pd.read_csv(ROOT / "results" / "ala_scan.csv")[["position", "residue"]]
d = full.merge(ala, on="position")
d["hydro"] = d.residue.map(KD)
d["volume"] = d.residue.map(VOL)
d["charge"] = d.residue.map(lambda a: abs(CHG.get(a, 0)))
d["burial"] = 1 - d.rsa  # packing density proxy

rows = []
for split_name, group_col in (("hotspot vs non-hotspot (g_i >= 2 kcal/mol)", "is_hotspot"),
                              ("in-patch vs outside-the-patch (spatially clustered hotspots only)", "in_patch")):
    a = d[d[group_col].astype(bool)]
    b = d[~d[group_col].astype(bool)]
    print(f"\n=== {split_name}: n={len(a)} vs n={len(b)} ===")
    for prop in ("hydro", "volume", "charge", "burial"):
        u, p = mannwhitneyu(a[prop], b[prop], alternative="two-sided")
        auc = u / (len(a) * len(b))
        rows.append(dict(split=split_name, property=prop, mean_group1=round(a[prop].mean(), 3),
                         mean_group2=round(b[prop].mean(), 3), mannwhitney_p=round(p, 5), AUC=round(auc, 3)))
        print(f"  {prop:8s} mean(group1)={a[prop].mean():+.3f}  mean(group2)={b[prop].mean():+.3f}  "
              f"Mann-Whitney p={p:.4g}  AUC={auc:.3f}")

# amino-acid composition, chi-square
print("\n=== amino-acid composition, hotspot vs non-hotspot ===")
ct = pd.crosstab(d.residue, d.is_hotspot)
chi2, p_chi, _, _ = chi2_contingency(ct)
print(f"chi-square test of composition difference: chi2={chi2:.2f}, p={p_chi:.4g}, df={ct.shape[0]-1}")
comp = (d.groupby("is_hotspot").residue.value_counts(normalize=True).rename("frac").reset_index())
comp.to_csv(HERE / "resultsH_composition_by_residue.csv", index=False)

pd.DataFrame(rows).to_csv(HERE / "resultsH_hotspot_composition.csv", index=False)
pd.DataFrame([dict(test="amino-acid composition chi-square", chi2=round(chi2, 3), p=round(p_chi, 4))]).to_csv(
    HERE / "resultsH_composition_chisq.csv", index=False)
print("\nsaved resultsH_hotspot_composition.csv, resultsH_composition_by_residue.csv, resultsH_composition_chisq.csv")
