"""
ROAD B - step B4 (tidy): turn the raw FoldX AlaScan output into results/ala_scan.csv
for CHAIN A only.

The raw file lists every residue of all 4 chains in PDB order, like:
    SER 96 to ALA energy change is 0.00924364
There is NO chain column, but chain A is simply the FIRST block: residues run
96..289, then the numbering RESETS to 96 when chain B begins. So we read lines
until the position stops increasing, and keep that first block = chain A.

Output columns (as the spec asks): position, residue, ddG
  - position = residue number
  - residue  = wild-type amino acid (1-letter)
  - ddG      = FoldX energy change on mutating that residue to alanine (kcal/mol);
               a BIG positive value = load-bearing residue = hotspot candidate.
"""
import re
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CHAIN = cfg["structure"]["chain"]

RAW = ROOT / "results" / "foldx_raw" / "tp53_repaired_AS.fxout"
OUT = ROOT / "results" / "ala_scan.csv"

three_to_one = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    # FoldX writes histidine tautomers with digit codes -> all are histidine:
    "H1S":"H","H2S":"H","H3S":"H",
}

# residue token may contain a digit (FoldX histidines: H1S/H2S/H3S)
line_re = re.compile(r"^([A-Z][A-Z0-9]{2})\s+(\d+)\s+to ALA energy change is\s+([-\d.eE]+)")

rows = []
prev_pos = None
for raw_line in RAW.read_text().splitlines():
    m = line_re.match(raw_line.strip())
    if not m:
        continue
    resn, pos, ddg = m.group(1), int(m.group(2)), float(m.group(3))
    # stop at the first chain boundary (numbering resets / stops increasing)
    if prev_pos is not None and pos <= prev_pos:
        break
    prev_pos = pos
    rows.append({"position": pos,
                 "residue": three_to_one.get(resn, "X"),
                 "ddG": round(ddg, 4)})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Chain used: {CHAIN} (first block of the AlaScan file)")
print(f"Residues in ala_scan.csv: {len(df)}  (positions {df.position.min()}..{df.position.max()})")
print(f"Saved -> {OUT.relative_to(ROOT)}")
print("\n=== first 12 rows ===")
print(df.head(12).to_string(index=False))
print("\n=== ddG summary (kcal/mol) ===")
print(df["ddG"].describe().round(3).to_string())
print("\n=== top 10 highest ddG (strongest hotspot candidates) ===")
print(df.sort_values("ddG", ascending=False).head(10).to_string(index=False))
