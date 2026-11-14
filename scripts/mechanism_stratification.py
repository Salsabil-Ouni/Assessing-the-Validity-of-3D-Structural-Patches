"""
Mechanism stratification: structural vs contact mutants (Chapter 2, Chapter 5
Discussion). Chapter 2 names only six literature codons (structural: R175,
G245, R249, R282; contact: R248, R273) - too few to power a standalone test.
This script does three things:

  (a) A descriptive case-study table for exactly those six named residues
      (illustrative, n too small for a formal test - reported as such).
  (b) A properly powered version: DNA-proximity (dna_proximity.py, an
      independent structural axis from the DNA-bound 1TUP structure) is used
      to classify chain-A residues as near-DNA (proxy: contact-type) or
      far-from-DNA (proxy: structural-type), and Q1/Q2 are re-run within each
      subset on both the experimental (n=37) and functional (n=175,194) data.
  (c) A sensitivity sweep of the near/far cutoff itself (5/8/10 A), since
      every other threshold in this pipeline is swept and this one, being new
      in this appendix, should be too (results/mechanism_sensitivity.csv).

This is an EXPLORATORY appendix analysis, not part of the pre-registered main
pipeline: unlike config.yaml's six thresholds, the cutoffs here were not fixed
before the data were inspected, and roughly a dozen tests are run across the
two datasets/mechanism subsets/cutoffs without multiple-comparison correction.
Results should be read as suggestive, not confirmatory (see appendix text).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FE = ROOT / "functional_ext"
DNA_CUT_DEFAULT = 8.0
SWEEP_CUTOFFS = [5.0, 8.0, 10.0]

dna = pd.read_csv(RES / "dna_proximity.csv")
full = pd.read_csv(RES / "residue_table_full_cA.csv")
exp = pd.read_csv(RES / "residue_table_cA.csv")

# ---- (a) descriptive case study: the six literature-named codons -----------
canon = {175: "structural", 245: "structural", 249: "structural", 282: "structural",
         248: "contact", 273: "contact"}
case = full[full.position.isin(canon)].merge(dna[["position", "min_dist_dna"]], on="position")
case["mechanism"] = case.position.map(canon)
case = case.merge(exp[["position", "rep_ddg", "destabilizing"]], on="position", how="left")
print("=== (a) Case study: the six literature-named hotspot codons (n=6, illustrative only) ===")
print(case[["position", "mechanism", "ala_ddg", "is_hotspot", "in_patch", "rsa",
            "min_dist_dna", "rep_ddg", "destabilizing"]].sort_values("mechanism").to_string(index=False))
print("\nn=6 is far too small for a statistical test; shown as a targeted check of the direction")
print("Chapter 2 predicts (structural codons buried/hotspot, contact codons not), not as evidence.\n")

# ---- shared helpers ---------------------------------------------------------
def fisher(df):
    a = int((df.destabilizing & df.in_patch).sum()); b = int((df.destabilizing & ~df.in_patch).sum())
    c = int((~df.destabilizing & df.in_patch).sum()); d = int((~df.destabilizing & ~df.in_patch).sum())
    if min(a + b, c + d, a + c, b + d) == 0:
        return (a, b, c, d), None, None
    o = odds_ratio([[a, b], [c, d]]); ci = o.confidence_interval(0.95)
    _, p = fisher_exact([[a, b], [c, d]])
    stat = o.statistic if np.isfinite(o.statistic) else None
    return (a, b, c, d), stat, p

three1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T',
'Trp':'W','Tyr':'Y','Val':'V'}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')
HOTSPOTS = [('R',175,'H'),('R',248,'W'),('R',248,'Q'),('R',249,'S'),
            ('G',245,'S'),('R',273,'H'),('R',282,'W')]

def load(fname):
    df = pd.read_csv(FE / "data" / fname)
    rows = []
    for _, r in df.iterrows():
        m = RX.match(str(r['hgvs_pro']))
        if not m:
            continue
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if wt in three1 and mut in three1 and wt != mut and pd.notna(r['score']):
            rows.append((pos, three1[wt], three1[mut], float(r['score'])))
    return pd.DataFrame(rows, columns=['position', 'wt', 'mut', 'score'])

def orient(v):
    hs = pd.concat([v[(v.position == p) & (v.wt == w) & (v.mut == m)] for w, p, m in HOTSPOTS])
    if len(hs) == 0:
        return v
    if hs.score.mean() < v.score.mean():
        v = v.copy(); v["score"] = -v["score"]
    return v

giac = orient(load("giacomelli_scores.csv")).groupby("position", as_index=False).agg(damage=("score", "mean"))
kato = orient(load("kato_scores.csv")).groupby("position", as_index=False).agg(damage=("score", "mean"))
giac = giac.merge(full[["position", "patch_score", "rsa"]], on="position")
kato = kato.merge(full[["position", "patch_score", "rsa"]], on="position")

# ---- (b) + (c) sweep the near/far DNA cutoff --------------------------------
n_tests_run = 0
sweep_rows = []
for cut in SWEEP_CUTOFFS:
    dna_c = dna.copy()
    dna_c["near_dna"] = dna_c.min_dist_dna < cut

    exp_dna = exp.merge(dna_c[["position", "near_dna"]], on="position", how="inner")
    row = {"dna_cutoff": cut}
    for label, sub in [("far", exp_dna[~exp_dna.near_dna]), ("near", exp_dna[exp_dna.near_dna])]:
        tab, orr, p = fisher(sub)
        n_tests_run += 1
        row[f"exp_{label}_n"] = len(sub)
        row[f"exp_{label}_OR"] = None if orr is None else round(orr, 2)
        row[f"exp_{label}_p"] = None if p is None else round(p, 4)

    for name, d in [("giac", giac), ("kato", kato)]:
        dd = d.merge(dna_c[["position", "near_dna"]], on="position")
        for label, sub in [("far", dd[~dd.near_dna]), ("near", dd[dd.near_dna])]:
            n_tests_run += 1
            if len(sub) < 5:
                row[f"{name}_{label}_n"] = len(sub)
                row[f"{name}_{label}_rho"] = None
                row[f"{name}_{label}_p"] = None
                continue
            rho, p = spearmanr(sub.damage, sub.patch_score)
            row[f"{name}_{label}_n"] = len(sub)
            row[f"{name}_{label}_rho"] = round(rho, 3)
            row[f"{name}_{label}_p"] = round(p, 5)
    sweep_rows.append(row)

sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(RES / "mechanism_sensitivity.csv", index=False)

print(f"=== (b)/(c) DNA-proximity-stratified Q1 (experimental) across cutoffs {SWEEP_CUTOFFS} A ===")
for r in sweep_rows:
    print(f"  cutoff={r['dna_cutoff']}: far n={r['exp_far_n']} OR={r['exp_far_OR']} p={r['exp_far_p']}"
          f"  |  near n={r['exp_near_n']} OR={r['exp_near_OR']} p={r['exp_near_p']}")

print(f"\n=== (b)/(c) DNA-proximity-stratified Q2 (functional replication) across cutoffs {SWEEP_CUTOFFS} A ===")
for name in ["giac", "kato"]:
    label = "Giacomelli 2018" if name == "giac" else "Kato 2003"
    print(f"  {label}:")
    for r in sweep_rows:
        print(f"    cutoff={r['dna_cutoff']}: far n={r[f'{name}_far_n']} rho={r[f'{name}_far_rho']} p={r[f'{name}_far_p']}"
              f"  |  near n={r[f'{name}_near_n']} rho={r[f'{name}_near_rho']} p={r[f'{name}_near_p']}")

print(f"\nTotal stratified tests run across the {len(SWEEP_CUTOFFS)}-cutoff sweep: {n_tests_run}")
print("No multiple-comparison correction applied (exploratory analysis; see appendix caveat).")
print("\nSaved -> results/mechanism_sensitivity.csv")
