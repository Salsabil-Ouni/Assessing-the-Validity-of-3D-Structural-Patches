"""
REPLICATION of the microenvironment effect across THREE independent TP53 functional
assays (all separate from the 37-residue stability result):

  Giacomelli 2018 (drug-selection saturation scan)   urn:mavedb:00000068-0-1
  Kotler 2018     (cell-growth fitness scan)         urn:mavedb:00000059-a-1
  Kato 2003       (yeast transcriptional activity)   urn:mavedb:00001234-0-1

Each assay has its own sign convention, so scores are AUTO-ORIENTED using known
loss-of-function hotspots (R175H, R248W/Q, R249S, G245S, R273H, R282W) so that
HIGHER = MORE DAMAGING in every dataset. Then the identical analysis is run:

  Q2 (primary, cutoff-free): Spearman(damage, patch_score); full + buried-only
  Q2 RSA-adjusted OLS: does the patch effect survive burial?
  Q1: enrichment with 'damaging' = top tercile (a relative cutoff, comparable across assays)
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "functional_ext"
RSA_CUT = 0.10
three1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G',
'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T',
'Trp':'W','Tyr':'Y','Val':'V'}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')
HOTSPOTS = [('R',175,'H'),('R',248,'W'),('R',248,'Q'),('R',249,'S'),
            ('G',245,'S'),('R',273,'H'),('R',282,'W')]

struct = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position","patch_score","rsa","in_patch"]]

def load(fname):
    df = pd.read_csv(FE / "data" / fname)
    rows=[]
    for _,r in df.iterrows():
        m = RX.match(str(r['hgvs_pro']))
        if not m: continue
        wt,pos,mut = m.group(1), int(m.group(2)), m.group(3)
        if wt in three1 and mut in three1 and wt!=mut and pd.notna(r['score']):
            rows.append((pos, three1[wt], three1[mut], float(r['score'])))
    return pd.DataFrame(rows, columns=['position','wt','mut','score'])

def orient(v):
    """Flip so HIGHER = MORE DAMAGING, decided by known LOF hotspots."""
    hs = pd.concat([v[(v.position==p)&(v.wt==w)&(v.mut==m)] for w,p,m in HOTSPOTS])
    if len(hs)==0: return v, "unknown (no hotspots found)"
    if hs.score.mean() < v.score.mean():
        v = v.copy(); v["score"] = -v["score"]
        return v, f"FLIPPED (hotspots scored low: {hs.score.mean():.2f} vs mean {v.score.mean()*-1:.2f})"
    return v, f"kept (hotspots scored high: {hs.score.mean():.2f} vs mean {v.score.mean():.2f})"

def analyse(name, fname):
    v = load(fname)
    v, note = orient(v)
    res = v.groupby("position", as_index=False).agg(damage=("score","mean"), n_sub=("score","size"))
    d = res.merge(struct, on="position", how="inner").dropna(subset=["rsa"])
    cut = d.damage.quantile(2/3)                 # top tercile = 'damaging'
    d["damaging"] = d.damage > cut
    b = d[d.rsa < RSA_CUT]
    # Q2
    r_full,p_full = spearmanr(d.damage, d.patch_score)
    r_bur ,p_bur  = spearmanr(b.damage, b.patch_score) if len(b)>=5 else (np.nan,np.nan)
    # Q2 RSA-adjusted
    o1 = sm.OLS(d.damage, sm.add_constant(d[["patch_score"]])).fit()
    o2 = sm.OLS(d.damage, sm.add_constant(d[["patch_score","rsa"]])).fit()
    # Q1
    a=int((d.damaging&d.in_patch).sum()); bb=int((d.damaging&~d.in_patch).sum())
    c=int((~d.damaging&d.in_patch).sum()); dd=int((~d.damaging&~d.in_patch).sum())
    if min(a+bb,c+dd,a+c,bb+dd)==0: orr,pf=np.nan,np.nan
    else:
        oo=odds_ratio([[a,bb],[c,dd]]); orr=oo.statistic; _,pf=fisher_exact([[a,bb],[c,dd]])
    print(f"\n### {name}  (orientation: {note})")
    print(f"  residues overlapping structure: {len(d)}  | buried(RSA<{RSA_CUT}): {len(b)}")
    print(f"  Q1 enrichment (top-tercile damaging): OR={orr:.2f} p={pf:.3g}   table=({a},{bb},{c},{dd})")
    print(f"  Q2 Spearman(damage, patch_score): FULL rho={r_full:+.3f} p={p_full:.3g} | BURIED rho={r_bur:+.3f} p={p_bur:.3g}")
    print(f"  Q2 OLS patch coef: unadj={o1.params.iloc[1]:.4f}(p={o1.pvalues.iloc[1]:.3g}) | "
          f"RSA-adj={o2.params.iloc[1]:.4f}(p={o2.pvalues.iloc[1]:.3g})")
    return dict(assay=name, n=len(d), n_buried=len(b), Q1_OR=orr, Q1_p=pf,
                Q2_rho=r_full, Q2_p=p_full, Q2_buried_rho=r_bur, Q2_buried_p=p_bur,
                patch_coef_unadj=o1.params.iloc[1], patch_p_unadj=o1.pvalues.iloc[1],
                patch_coef_rsaadj=o2.params.iloc[1], patch_p_rsaadj=o2.pvalues.iloc[1])

out=[]
out.append(analyse("Giacomelli 2018 (drug selection)", "giacomelli_scores.csv"))
out.append(analyse("Kotler 2018 (cell growth)",        "kotler_scores.csv"))
out.append(analyse("Kato 2003 (transcriptional)",      "kato_scores.csv"))
t=pd.DataFrame(out); t.to_csv(FE/"results"/"replication_table.csv", index=False)
print("\n================ REPLICATION SUMMARY ================")
print(t[["assay","n","Q2_rho","Q2_p","patch_p_rsaadj","Q2_buried_rho","Q2_buried_p"]].to_string(index=False))
print("\nSaved -> functional_ext/results/replication_table.csv")
