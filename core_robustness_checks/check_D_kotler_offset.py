"""Check D: offset search for the Kotler et al. dataset's numbering."""
import re
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

three1 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
          "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')

fa = urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/P04637.fasta").read().decode()
full_seq = "".join(fa.splitlines()[1:])

df = pd.read_csv(ROOT / "functional_ext" / "data" / "kotler_scores.csv")
wt_by_pos = {}
for h in df.hgvs_pro.dropna().astype(str):
    m = RX.match(h)
    if m and m.group(1) in three1:
        wt_by_pos.setdefault(int(m.group(2)), three1[m.group(1)])
print(f"Kotler: {len(wt_by_pos)} distinct positions with a parseable wild-type residue")

rows = []
best = (-1, None)
for off in range(-100, 101):
    n_ok = n_tot = 0
    for pos, wt in wt_by_pos.items():
        target = pos + off
        if 1 <= target <= len(full_seq):
            n_tot += 1
            n_ok += full_seq[target - 1] == wt
    if n_tot == 0:
        continue
    pct = 100 * n_ok / n_tot
    rows.append(dict(offset=off, n_checked=n_tot, n_match=n_ok, pct_match=round(pct, 1)))
    if pct > best[0]:
        best = (pct, off)

res = pd.DataFrame(rows).sort_values("pct_match", ascending=False)
res.to_csv(HERE / "resultsD_kotler_offset.csv", index=False)
print(res.head(10).to_string(index=False))
print(f"\nBest offset: {best[1]:+d}, {best[0]:.1f}% concordance (threshold used to retain a dataset elsewhere: >95%).")
print(f"At offset 0 (as originally checked): {res[res.offset==0].pct_match.iloc[0]:.1f}%")
