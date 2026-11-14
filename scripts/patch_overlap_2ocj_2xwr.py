"""
Do 2OCJ and 2XWR agree on WHICH residues form the patch, not just on the summary
statistics (Q1 OR, Q2 rho)? Two structures could produce similar aggregate numbers
while disagreeing on the actual residues involved; this checks that directly.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

ocj = pd.read_csv(RES / "residue_table_full_cA.csv")[["position", "is_hotspot", "in_patch"]]
xwr = pd.read_csv(RES / "residue_table_2xwr.csv")[["position", "is_hotspot", "in_patch"]]

m = ocj.merge(xwr, on="position", suffixes=("_ocj", "_xwr"), how="inner")
print(f"Shared positions: {len(m)}")

def overlap(col):
    a = set(m.loc[m[f"{col}_ocj"], "position"])
    b = set(m.loc[m[f"{col}_xwr"], "position"])
    inter, union = a & b, a | b
    jaccard = len(inter) / len(union) if union else float("nan")
    # agreement rate = fraction of ALL shared positions where the two structures agree (both True or both False)
    agree = (m[f"{col}_ocj"] == m[f"{col}_xwr"]).mean()
    return a, b, inter, union, jaccard, agree

for col, label in [("is_hotspot", "Hotspot"), ("in_patch", "In-patch")]:
    a, b, inter, union, jaccard, agree = overlap(col)
    print(f"\n=== {label} ===")
    print(f"  2OCJ: n={len(a)}  2XWR: n={len(b)}  intersection={len(inter)}  union={len(union)}")
    print(f"  Jaccard index = {jaccard:.3f}")
    print(f"  Position-level agreement (both True or both False) = {agree:.3f} "
          f"({int(agree*len(m))}/{len(m)})")
    only_ocj = a - b
    only_xwr = b - a
    print(f"  {label} only in 2OCJ ({len(only_ocj)}): {sorted(only_ocj)}")
    print(f"  {label} only in 2XWR ({len(only_xwr)}): {sorted(only_xwr)}")

# canonical hotspot codons + zinc ligands: do they agree specifically?
canon = {175: "structural", 245: "structural", 249: "structural", 282: "structural",
         248: "contact", 273: "contact", 176: "zinc", 179: "zinc", 238: "zinc", 242: "zinc"}
print("\n=== Canonical positions: in_patch agreement ===")
chk = m[m.position.isin(canon)].copy()
chk["role"] = chk.position.map(canon)
print(chk[["position", "role", "in_patch_ocj", "in_patch_xwr"]].sort_values("position").to_string(index=False))

pd.DataFrame([{
    "n_shared_positions": len(m),
    "hotspot_jaccard": round(overlap("is_hotspot")[4], 3),
    "hotspot_agreement": round(overlap("is_hotspot")[5], 3),
    "inpatch_jaccard": round(overlap("in_patch")[4], 3),
    "inpatch_agreement": round(overlap("in_patch")[5], 3),
}]).to_csv(RES / "patch_overlap_2ocj_2xwr.csv", index=False)
print("\nSaved -> results/patch_overlap_2ocj_2xwr.csv")
