"""Check C: Q2 with the four zinc-coordinating residues excluded."""
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
ZINC = [176, 179, 238, 242]  # Cys176, His179, Cys238, Cys242

exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")

print("Zinc ligands present in n=37 set:", [p for p in ZINC if p in exp.position.values])
print("Zinc ligands present in n=194 set:", [p for p in ZINC if p in full.position.values])

rows = []
for label, tab, outcome, score_cols in (
        ("n=37 (experimental)", exp, "abs_rep_ddg", [("patch score", "patch_score"), ("own alanine-scan value", None)]),
        ("n=194 (FoldX-extended)", full, "ala_ddg", [("patch score", "patch_score"), ("own alanine-scan value", "ala_ddg")])):
    for name, col in score_cols:
        if col is None:
            continue
        for suffix, sub in (("all residues", tab), ("zinc ligands excluded", tab[~tab.position.isin(ZINC)])):
            if col == outcome:  # own-value vs own-value is trivial; skip
                continue
            rho, p = spearmanr(sub[col], sub[outcome])
            rows.append(dict(dataset=label, score=name, subset=suffix, n=len(sub), spearman=round(rho, 4), p=round(p, 4)))
            print(f"{label:26s} | {name:24s} | {suffix:22s} | n={len(sub):3d} | rho={rho:+.3f} (p={p:.3g})")

pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent / "resultsC_zinc_leverage.csv", index=False)
print("\nsaved resultsC_zinc_leverage.csv")
