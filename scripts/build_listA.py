"""
ROAD A  -  build List A: experimentally-measured destabilising mutations.

Reads ThermoMutDB JSON (downloaded as-is) and produces results/listA_variants.csv.
Every setting comes from config.yaml. This script NEVER looks at the structure.

Steps (matching the project spec):
  A2  keep only single-point mutations whose position is inside the region of interest
  A3  standardise names to the form R175H and add a numeric position column
  A4  collapse duplicate measurements of the same mutation by taking the MEAN ddG
  A5  add a 'label' column using the ddG cutoffs from config.yaml
  A6  save results/listA_variants.csv
"""
import json, re, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))

REGION_START = cfg["protein"]["region_start"]
REGION_END   = cfg["protein"]["region_end"]
DEL_BELOW    = cfg["cutoffs"]["ddg_deleterious_below"]   # ddG < this  -> deleterious
STAB_ABOVE   = cfg["cutoffs"]["ddg_stabilising_above"]   # ddG > this  -> stabilising

JSON_PATH = ROOT / "data" / "thermomutdb.json"
OUT_PATH  = ROOT / "results" / "listA_variants.csv"

# ---- load the full database, keep only p53 records -------------------------
records = json.load(open(JSON_PATH, encoding="utf-8"))
p53 = [r for r in records if r.get("protein") == "Cellular tumor antigen p53"]

# A2 + A3: keep single-point mutations like "R175H", parse wt / position / mut
single_re = re.compile(r"^([A-Z])(\d+)([A-Z])$")
parsed = []
for r in p53:
    code = (r.get("mutation_code") or "").strip()
    m = single_re.match(code)
    if not m:
        continue                                   # drop multi-mutations / odd formats
    ddg = r.get("ddg")
    if not isinstance(ddg, (int, float)):
        continue                                   # drop rows without a numeric ddG
    wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    if not (REGION_START <= pos <= REGION_END):
        continue                                   # keep only the region of interest
    parsed.append({"mutation": f"{wt}{pos}{mut}", "wt": wt,
                   "position": pos, "mut": mut, "ddg": float(ddg)})

df = pd.DataFrame(parsed)

# A4: collapse duplicates (same mutation measured by >1 study) -> mean ddG
agg = (df.groupby(["mutation", "wt", "position", "mut"], as_index=False)
         .agg(ddg=("ddg", "mean"), n_measurements=("ddg", "size")))

# A5: label using the config cutoffs (remember: negative ddG = destabilising here)
def label(d):
    if d < DEL_BELOW:   return "deleterious"
    if d > STAB_ABOVE:  return "stabilising"
    return "neutral"
agg["label"] = agg["ddg"].apply(label)

# tidy: sort by position, round ddG for readability
agg = agg.sort_values("position").reset_index(drop=True)
agg["ddg"] = agg["ddg"].round(3)

# A6: save
agg.to_csv(OUT_PATH, index=False)

# ---- report to the screen --------------------------------------------------
print(f"Region kept: {REGION_START}-{REGION_END}")
print(f"Cutoffs: deleterious if ddG < {DEL_BELOW} ; stabilising if ddG > {STAB_ABOVE}")
print(f"Unique single-point mutations in region: {len(agg)}")
print(f"\nSaved -> {OUT_PATH.relative_to(ROOT)}")
print("\n=== First 10 rows ===")
print(agg.head(10).to_string(index=False))
print("\n=== Count per label ===")
print(agg["label"].value_counts().to_string())
