"""
ROAD B - step B5: group hotspot residues into 3D PATCHES.

A hotspot = chain-A residue with AlaScan ddG >= cutoff (config: ala_hotspot_ddg).
Two hotspots are "neighbours" if their closest HEAVY atoms (non-hydrogen) are
within patch_distance_angstrom (config). Build a graph of neighbours; each
connected component is a patch.

Outputs:
  results/listB_patches.csv      patch_id, n_residues, residues, patch_score(sum ddG)
  results/patch_membership.csv   position, in_patch   (for ALL chain-A residues)

NOTE: this script never looks at List A (Road A) data.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CHAIN   = cfg["structure"]["chain"]
CUTOFF  = cfg["cutoffs"]["ala_hotspot_ddg"]
# Patch neighbour definition for THIS script only: Ca-Ca distance <= 6 Angstrom,
# matching the supervisor's hotspotPatches.py. This intentionally overrides
# config.yaml's patch_distance_angstrom (8.0 A, closest-heavy-atom), which other
# scripts still use for the microenvironment score and is left untouched.
DIST    = 6.0
PDB     = ROOT / cfg["structure"]["repaired_pdb"]

ala = pd.read_csv(ROOT / "results" / "ala_scan.csv")
hot = ala[ala["ddG"] >= CUTOFF].copy()
hot_positions = set(hot["position"])
ddg_of = dict(zip(hot["position"], hot["ddG"]))
print(f"Hotspot cutoff: ddG >= {CUTOFF}  ->  {len(hot_positions)} hotspot residues")
print(f"Neighbour distance: closest heavy atoms within {DIST} Angstrom")

# ---- gather Ca coordinates per hotspot residue (chain A) ------------------
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
ca_coord = {}   # position -> Ca coordinate
for res in chain:
    pos = res.id[1]
    if pos in hot_positions and res.has_id("CA"):
        ca_coord[pos] = res["CA"].coord
heavy = ca_coord

def min_dist(p, q):
    # Ca-Ca distance, matching the supervisor's hotspotPatches.py
    return float(np.linalg.norm(ca_coord[p] - ca_coord[q]))

# ---- build neighbour graph -------------------------------------------------
G = nx.Graph()
G.add_nodes_from(heavy.keys())
for p, q in itertools.combinations(sorted(heavy), 2):
    if min_dist(p, q) <= DIST:
        G.add_edge(p, q)

components = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"\nConnected components (patches) found: {len(components)}")
sizes = [len(c) for c in components]
print("Component sizes:", sizes)
print(f"  -> components with >=2 residues: {sum(s>=2 for s in sizes)}")
print(f"  -> singletons (isolated hotspots): {sum(s==1 for s in sizes)}")

# ---- write listB_patches.csv ----------------------------------------------
patch_rows = []
for i, comp in enumerate(components, start=1):
    members = sorted(comp)
    score = round(sum(ddg_of[m] for m in members), 3)
    patch_rows.append({"patch_id": i,
                       "n_residues": len(members),
                       "residues": ",".join(map(str, members)),
                       "patch_score": score})
patchesdf = pd.DataFrame(patch_rows)
patchesdf.to_csv(ROOT / "results" / "listB_patches.csv", index=False)

print("\n=== listB_patches.csv ===")
print(patchesdf.to_string(index=False))

# ---- write patch_membership.csv (ALL chain-A residues) --------------------
# in_patch = residue belongs to a real PATCH (connected component of size >= 2).
# Isolated single hotspots (size-1 components) are NOT counted as a patch.
in_patch_positions = set()
for comp in components:
    if len(comp) >= 2:
        in_patch_positions |= comp

all_positions = sorted(ala["position"])
member = pd.DataFrame({"position": all_positions})
member["in_patch"] = member["position"].isin(in_patch_positions)
member.to_csv(ROOT / "results" / "patch_membership.csv", index=False)

print("\n=== patch_membership.csv summary ===")
print(f"residues total: {len(member)}")
print(f"in_patch = True : {member['in_patch'].sum()}")
print(f"in_patch = False: {(~member['in_patch']).sum()}")
