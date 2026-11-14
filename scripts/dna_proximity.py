"""
Per-residue proximity to DNA, computed from 1TUP (p53 DNA-binding domain trimer
bound to its response element, chain A protein + chains E/F DNA;
data/candidates/1TUP.pdb - the same DNA-bound structure rejected for the main
patch-construction pipeline because DNA distorts the loops, Section
structure-prioritisation). Used ONLY here, as an independent structural axis
distinct from solvent burial, to (a) operationalise the structural-vs-contact
mutation distinction (Chapter 2) beyond the six named literature residues, and
(b) test whether it predicts functional impact independently of burial
(interface_beyond_burial.py).

Numbering check: 1TUP chain A residue identities at all six canonical hotspot
positions (175/245/248/249/273/282) and all four zinc ligands (176/179/238/242)
match the human P04637 sequence used throughout this project (verified before
using this structure), so positions merge directly onto the 2OCJ-based tables
by residue number - no renumbering needed.

Output: results/dna_proximity.csv (position, min_dist_dna, dna_contact @ 5A).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "data" / "candidates" / "1TUP.pdb"
CONTACT_CUTOFF = 5.0  # Angstrom, canonical direct-contact distance in structural biology

structure = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]
dna_chains = [structure[c] for c in ("E", "F")]

dna_heavy = []
for ch in dna_chains:
    for res in ch:
        if res.resname in ("DA", "DT", "DG", "DC", "A", "T", "G", "C"):
            dna_heavy.extend(a.coord for a in res if (a.element or "").strip().upper() != "H")
dna_heavy = np.array(dna_heavy)
print(f"DNA heavy atoms (chains E+F): {len(dna_heavy)}")

# 1TUP is a trimer (chains A, B, C) bound to a consensus response element; only
# the copies actually engaging a well-resolved DNA half-site give a meaningful
# distance; a copy sitting away from any half-site would wrongly read as "far
# from DNA" for every one of its residues. Checked directly: chain A's known
# contact residues (R248, R273, K120, S241, A276, C277, R280) all read
# > 10 A, while chains B and C read < 5 A at the same positions - so chain A is
# the poorly-engaged copy in this asymmetric unit. Taking the minimum distance
# across all three protein chains per residue number avoids this per-chain
# engagement artefact and reflects the best-contacting copy.
protein_chains = [structure[c] for c in ("A", "B", "C")]
by_pos = {}
for ch in protein_chains:
    for res in ch:
        if res.id[0] != " ":
            continue
        pos = res.id[1]
        heavy = np.array([a.coord for a in res if (a.element or "").strip().upper() != "H"])
        if len(heavy) == 0:
            continue
        d = float(np.sqrt(((heavy[:, None, :] - dna_heavy[None, :, :]) ** 2).sum(-1)).min())
        by_pos[pos] = min(d, by_pos.get(pos, np.inf))

out = pd.DataFrame([{"position": p, "min_dist_dna": round(d, 2)} for p, d in by_pos.items()]).sort_values("position")
out["dna_contact"] = out.min_dist_dna < CONTACT_CUTOFF
out.to_csv(ROOT / "results" / "dna_proximity.csv", index=False)

print(f"\nResidues: {len(out)}  |  direct DNA contacts (<{CONTACT_CUTOFF} A): {out.dna_contact.sum()}")
print("\nCanonical hotspot codons (Chapter 2) - sanity check against this metric:")
canon = {175: "structural", 245: "structural", 249: "structural", 282: "structural",
         248: "contact", 273: "contact"}
chk = out[out.position.isin(canon)].copy()
chk["literature_class"] = chk.position.map(canon)
print(chk.sort_values("position").to_string(index=False))
print("\nSaved -> results/dna_proximity.csv")
