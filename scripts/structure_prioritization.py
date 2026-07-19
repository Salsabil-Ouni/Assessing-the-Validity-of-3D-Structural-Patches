"""
STRUCTURE PRIORITISATION (pipeline step).
Evaluate candidate TP53 DNA-binding-domain structures by:
  - experimental method + resolution
  - oligomeric state (n protein chains), WT vs engineered (from title)
  - coverage of residues 93-290 (best chain), internal gaps
  - presence of the structural zinc
Then rank and justify a selection.
"""
import re
from pathlib import Path
import requests
import pandas as pd
from Bio.PDB import PDBParser, is_aa

ROOT = Path(__file__).resolve().parents[1]
CAND = ["2OCJ", "1TUP", "1UOL", "2XWR", "4HJE", "2FEJ", "3Q05"]   # user list + 2 extra WT refs
REGION = range(93, 291)
SPAN = len(REGION)
ddir = ROOT / "data" / "candidates"; ddir.mkdir(parents=True, exist_ok=True)

def fetch(pid):
    f = ddir / f"{pid}.pdb"
    if not f.exists():
        r = requests.get(f"https://files.rcsb.org/download/{pid}.pdb", timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
    return f.read_text(errors="ignore")

rows = []
for pid in CAND:
    try:
        txt = fetch(pid)
    except Exception as e:
        rows.append({"pdb": pid, "note": f"download failed: {e}"}); continue
    method = next((l[10:].strip() for l in txt.splitlines() if l.startswith("EXPDTA")), "?")
    res = None
    for l in txt.splitlines():
        if l.startswith("REMARK   2 RESOLUTION"):
            m = re.search(r"([0-9]+\.[0-9]+)\s*ANGSTROM", l)
            if m: res = float(m.group(1))
    title = " ".join(l[10:].strip() for l in txt.splitlines() if l.startswith("TITLE"))
    model = PDBParser(QUIET=True).get_structure(pid, str(ddir / f"{pid}.pdb"))[0]
    # protein chains and their coverage of the region
    best = None
    nprot = 0
    for ch in model:
        nums = sorted({r.id[1] for r in ch if is_aa(r, standard=True)})
        if len(nums) < 30:
            continue
        nprot += 1
        in_reg = [n for n in nums if n in REGION]
        gaps = [x for x in range(min(in_reg), max(in_reg) + 1) if x not in set(nums)] if in_reg else []
        cov = len(in_reg)
        if best is None or cov > best["coverage"]:
            best = {"chain": ch.id, "coverage": cov,
                    "range": f"{min(nums)}-{max(nums)}" if nums else "-",
                    "gaps_in_region": len(gaps)}
    znt = sum(1 for ch in model for r in ch for a in r if (a.element or "").strip().upper() == "ZN")
    rows.append({"pdb": pid, "method": method, "resolution_A": res,
                 "n_protein_chains": nprot,
                 "best_chain": best["chain"] if best else "-",
                 "coverage_93_290": best["coverage"] if best else 0,
                 "pct_coverage": round(100 * best["coverage"] / SPAN, 1) if best else 0,
                 "gaps_in_region": best["gaps_in_region"] if best else "-",
                 "zinc": znt, "title": title[:70]})

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "structure_candidates.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(df.to_string(index=False))
