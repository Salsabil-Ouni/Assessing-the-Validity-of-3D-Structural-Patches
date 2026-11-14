"""
Multi-structure validation: does the hotspot/patch finding reproduce on a second,
independent WT p53 structure? This directly answers the open question on the
supervisor's own pipeline sketch ("are we going to use all the P53 WT structures
or not") that the main pipeline (built on 2OCJ alone) never revisited.

Structure chosen: 2XWR (1.68 A, WT, DNA-free/apo, residues 89-293, chain B
selected in the original prioritisation as its best chain). Rejected for the MAIN
analysis only because 2OCJ was already the supervisor-provided, FoldX-repaired file
(so prioritisation just confirmed it, rather than 2XWR being biologically inferior) --
2XWR is in fact HIGHER resolution than 2OCJ and, like it, WT and DNA-free, which is
exactly what makes it a clean independent check: same category of structure (WT,
apo), different crystal form, no DNA-induced distortion confound (unlike 1TUP/4HJE,
already used elsewhere in this project specifically because they are DNA-bound).

Numbering verified directly (see main() output) at the four zinc ligands and six
canonical hotspot codons before use, as for every external structure in this
pipeline.

Step 1 (this script, part A): extract chain B + its zinc, single altconf, renamed
to chain A for compatibility with downstream scripts, written to
structure/tp53_2xwr_chainA_zinc.pdb.
"""
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "candidates" / "2XWR.pdb"
OUT = ROOT / "structure" / "tp53_2xwr_chainB_zinc.pdb"

three1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

class ChainBZinc(Select):
    def accept_chain(self, chain):
        return chain.id == "B"
    def accept_residue(self, res):
        if res.id[0] == " ":
            return True
        if res.resname == "ZN" and res.parent.id == "B":
            return True
        return False
    def accept_atom(self, atom):
        return (not atom.is_disordered()) or atom.get_altloc() in ("A", " ")

structure = PDBParser(QUIET=True).get_structure("2xwr", str(SRC))
io = PDBIO()
io.set_structure(structure)
io.save(str(OUT), ChainBZinc())
print(f"Wrote -> {OUT}")

# numbering sanity check: canonical hotspot codons + zinc ligands
chain = PDBParser(QUIET=True).get_structure("chk", str(OUT))[0]["B"]
canon = {175: "R", 176: "C", 179: "H", 238: "C", 242: "C", 245: "G",
         248: "R", 249: "R", 273: "R", 282: "R"}
print("\nNumbering check (position: expected wt -> found):")
found = {r.id[1]: three1.get(r.resname, "X") for r in chain if r.id[0] == " "}
ok = True
for pos, wt in canon.items():
    got = found.get(pos, "MISSING")
    match = "OK" if got == wt else "MISMATCH"
    if got != wt:
        ok = False
    print(f"  {pos}: expected {wt}, found {got}  [{match}]")
print(f"\nAll canonical positions match: {ok}")
print(f"Residues in extracted chain: {len(found)}, range {min(found)}-{max(found)}")
