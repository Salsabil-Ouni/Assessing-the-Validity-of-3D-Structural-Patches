"""
ROAD B - step B1: inspect the FoldX-repaired structure.
Reports (a) whether the structural ZINC ion is present, and
(b) the residue numbering range and chain(s). Saves a small summary to
results/structure_info.txt for later numbering checks.
"""
from pathlib import Path
import yaml
from Bio.PDB import PDBParser, is_aa

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
PDB = ROOT / cfg["structure"]["repaired_pdb"]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("p53", str(PDB))

lines = []
def say(s=""):
    print(s); lines.append(s)

say(f"Structure file: {cfg['structure']['repaired_pdb']}")
models = list(structure)
say(f"Number of models: {len(models)}")
model = models[0]

# ---- (a) ZINC check: look for any atom/residue named ZN -----------------
zinc_atoms = []
for chain in model:
    for res in chain:
        resname = res.get_resname().strip()
        for atom in res:
            el = (atom.element or "").strip().upper()
            if resname == "ZN" or el == "ZN" or atom.get_name().strip().upper() == "ZN":
                zinc_atoms.append((chain.id, resname, res.id, atom.get_name(), tuple(round(c,2) for c in atom.coord)))
say("\n=== (a) ZINC ION CHECK ===")
if zinc_atoms:
    say(f"ZINC PRESENT: found {len(zinc_atoms)} zinc atom(s).")
    for z in zinc_atoms:
        say(f"   chain {z[0]}  resname {z[1]}  resid {z[2]}  atom {z[3]}  coord {z[4]}")
else:
    say("*** ZINC NOT FOUND ***")

# ---- (b) residue numbering range + chains -------------------------------
say("\n=== (b) CHAINS and RESIDUE NUMBERING ===")
for chain in model:
    aa_res = [res for res in chain if is_aa(res, standard=True)]
    hetres = [res.get_resname().strip() for res in chain if not is_aa(res, standard=True)]
    if aa_res:
        nums = [res.id[1] for res in aa_res]
        say(f"chain {chain.id!r}: {len(aa_res)} amino-acid residues, "
            f"numbering {min(nums)} .. {max(nums)}")
    if hetres:
        say(f"           non-amino-acid groups in chain {chain.id!r}: {sorted(set(hetres))}")

# save summary
out = ROOT / "results" / "structure_info.txt"
out.write_text("\n".join(lines), encoding="utf-8")
say(f"\nSaved summary -> {out.relative_to(ROOT)}")
