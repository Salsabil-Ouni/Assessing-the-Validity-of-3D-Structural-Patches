"""
MERGE - step M1: verify List A (experiment) and the structure use the SAME
residue numbering. For every position present in both, the wild-type amino acid
must be identical. Prints matches, mismatches, and positions missing from the
structure. Does NOT merge or run any statistics.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A   = pd.read_csv(ROOT / "results" / "listA_variants.csv")     # has: position, wt
ala = pd.read_csv(ROOT / "results" / "ala_scan.csv")           # has: position, residue (struct wt)

struct_wt = dict(zip(ala["position"], ala["residue"]))

# one row per (position, experimental wt) - wt is constant per position in List A
posA = A[["position", "wt"]].drop_duplicates().sort_values("position")

rows = []
for _, r in posA.iterrows():
    p, wtA = int(r["position"]), r["wt"]
    wtS = struct_wt.get(p)
    if wtS is None:
        status = "NOT IN STRUCTURE"
    elif wtA == wtS:
        status = "match"
    else:
        status = "*** MISMATCH ***"
    rows.append({"position": p, "listA_wt": wtA, "structure_wt": wtS, "status": status})

chk = pd.DataFrame(rows)
n_match    = (chk.status == "match").sum()
n_mismatch = (chk.status == "*** MISMATCH ***").sum()
n_missing  = (chk.status == "NOT IN STRUCTURE").sum()

print(f"List A distinct positions: {len(chk)}")
print(f"  match            : {n_match}")
print(f"  MISMATCH         : {n_mismatch}")
print(f"  not in structure : {n_missing}")

print("\n=== Spot-check of famous p53 positions ===")
for fp in [134, 143, 175, 245, 248, 273, 282]:
    if fp in struct_wt:
        inA = chk[chk.position == fp]
        a = inA["listA_wt"].iloc[0] if len(inA) else "(not in List A)"
        print(f"  pos {fp}: structure={struct_wt[fp]}  listA={a}")

print("\n=== Full alignment table ===")
print(chk.to_string(index=False))
