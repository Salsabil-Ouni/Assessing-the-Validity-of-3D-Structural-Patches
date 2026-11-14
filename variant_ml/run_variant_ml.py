"""
Step 2 of the variant-level test: exactly the pre-specified analysis in PLAN.md.
Do not edit settings after seeing results; deviations must be logged in README.md.
"""
import re
import sys
for _m in ("dask", "dask.array", "dask.dataframe", "dask.distributed"):
    sys.modules[_m] = None   # optional dask import in lightgbm breaks on this pandas; unused here
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, rankdata
from Bio.Align import substitution_matrices

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FE = ROOT / "functional_ext" / "data"
rng = np.random.default_rng(0)

# ----------------------------- data -----------------------------
three1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H',
          'Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W',
          'Tyr':'Y','Val':'V'}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')
HOTSPOTS = [('R',175,'H'),('R',248,'W'),('R',248,'Q'),('R',249,'S'),('G',245,'S'),('R',273,'H'),('R',282,'W')]

def load(fname):
    df = pd.read_csv(FE / fname)
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
    if len(hs) and hs.score.mean() < v.score.mean():
        v = v.copy(); v["score"] = -v["score"]
    return v

full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position", "rsa", "patch_score", "ala_ddg"]]
full = full.rename(columns={"rsa": "burial_rsa", "ala_ddg": "own_value"})
esm = pd.read_csv(HERE / "esm_llr.csv")

KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,
          P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
VOL = dict(A=88.6,R=173.4,N=114.1,D=111.1,C=108.5,Q=143.8,E=138.4,G=60.1,H=153.2,I=166.7,L=166.7,K=168.6,
           M=162.9,F=189.9,P=112.7,S=89.0,T=116.1,W=227.8,Y=193.6,V=140.0)
CHG = dict(K=1, R=1, D=-1, E=-1)
B62 = substitution_matrices.load("BLOSUM62")

def build(fname):
    v = orient(load(fname))
    v = v[(v.position >= 96) & (v.position <= 289)]
    v = v.groupby(["position", "wt", "mut"], as_index=False).score.mean()
    v = v.merge(esm, on=["position", "wt", "mut"], how="inner").merge(full, on="position", how="inner")
    v["d_hydro"] = v.mut.map(KD) - v.wt.map(KD)
    v["d_volume"] = v.mut.map(VOL) - v.wt.map(VOL)
    v["d_charge"] = v.mut.map(lambda a: CHG.get(a, 0)) - v.wt.map(lambda a: CHG.get(a, 0))
    v["blosum62"] = [B62[w][m] for w, m in zip(v.wt, v.mut)]
    v["wt_hydro"] = v.wt.map(KD); v["mut_hydro"] = v.mut.map(KD)
    v["y"] = (rankdata(v.score) - 1) / (len(v) - 1)
    return v.reset_index(drop=True)

SUBST = ["esm_llr", "d_hydro", "d_volume", "d_charge", "blosum62", "wt_hydro", "mut_hydro"]
SETS = {"F1 substitution only": SUBST,
        "F2 + burial": SUBST + ["burial_rsa"],
        "F3 + burial + own value": SUBST + ["burial_rsa", "own_value"],
        "F4 + burial + patch": SUBST + ["burial_rsa", "patch_score"],
        "F5 + burial + own + patch": SUBST + ["burial_rsa", "own_value", "patch_score"],
        "F6 + patch only": SUBST + ["patch_score"]}
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0, n_jobs=4, verbose=-1)
N_REP, N_FOLD, N_BOOT = 5, 5, 2000

def fold_ids(positions, rep):
    r = np.random.default_rng(1000 + rep)
    up = r.permutation(np.unique(positions))
    fid = {p: i % N_FOLD for i, p in enumerate(up)}
    return np.array([fid[p] for p in positions])

def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

results, contrasts, importances = [], [], []
for assay, fname in (("Giacomelli 2018", "giacomelli_scores.csv"), ("Kato 2003", "kato_scores.csv")):
    d = build(fname)
    print(f"\n=== {assay}: {len(d)} substitutions at {d.position.nunique()} residues ===")
    y = d.y.values; pos = d.position.values
    oof = {n: np.zeros((N_REP, len(d))) for n in SETS}
    for rep in range(N_REP):
        fid = fold_ids(pos, rep)
        for k in range(N_FOLD):
            tr, te = fid != k, fid == k
            for n, cols in SETS.items():
                m = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, cols], y[tr])
                oof[n][rep, te] = m.predict(d.loc[te, cols])
    for n in SETS:
        rho = np.mean([spearmanr(oof[n][r], y)[0] for r in range(N_REP)])
        rr = np.mean([r2(y, oof[n][r]) for r in range(N_REP)])
        results.append(dict(assay=assay, features=n, oof_spearman=round(rho, 4), oof_R2=round(rr, 4),
                            n_rows=len(d), n_residues=d.position.nunique()))
        print(f"  {n:28s} OOF Spearman {rho:.3f}   R2 {rr:.3f}")

    # residue bootstrap for the two primary contrasts
    idx_by_pos = {p: np.where(pos == p)[0] for p in np.unique(pos)}
    upos = np.array(list(idx_by_pos))
    def boot_diff(a, b):
        out = []
        for _ in range(N_BOOT):
            sel = np.concatenate([idx_by_pos[p] for p in rng.choice(upos, len(upos))])
            da = np.mean([spearmanr(oof[a][r][sel], y[sel])[0] for r in range(N_REP)])
            db = np.mean([spearmanr(oof[b][r][sel], y[sel])[0] for r in range(N_REP)])
            out.append(da - db)
        return np.array(out)
    pt = {n: np.mean([spearmanr(oof[n][r], y)[0] for r in range(N_REP)]) for n in SETS}
    for label, a, b in (("Contrast 1: F5 vs F3 (patch added to substitution+burial+own)", "F5 + burial + own + patch", "F3 + burial + own value"),
                        ("Contrast 2: F4 vs F2 (patch added to substitution+burial)", "F4 + burial + patch", "F2 + burial")):
        bd = boot_diff(a, b)
        contrasts.append(dict(assay=assay, contrast=label, delta_spearman=round(pt[a] - pt[b], 4),
                              ci_low=round(np.percentile(bd, 2.5), 4), ci_high=round(np.percentile(bd, 97.5), 4)))
        print(f"  {label}: delta {pt[a]-pt[b]:+.4f}  95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}]")

    # grouped permutation importance for the full model (F5), held-out folds only
    cols = SETS["F5 + burial + own + patch"]
    blocks = {"ESM-2 zero-shot score": ["esm_llr"],
              "amino-acid chemistry": ["d_hydro", "d_volume", "d_charge", "blosum62", "wt_hydro", "mut_hydro"],
              "burial (RSA)": ["burial_rsa"], "own alanine-scan value": ["own_value"], "patch score": ["patch_score"]}
    drops = {b: [] for b in blocks}
    for rep in range(N_REP):
        fid = fold_ids(pos, rep)
        for k in range(N_FOLD):
            tr, te = fid != k, fid == k
            m = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, cols], y[tr])
            Xte = d.loc[te, cols].copy(); yte = y[te]; base = spearmanr(m.predict(Xte), yte)[0]
            pte = pos[te]; up = np.unique(pte)
            for b, bc in blocks.items():
                for _ in range(3):
                    Xp = Xte.copy()
                    if b in ("burial (RSA)", "own alanine-scan value", "patch score"):
                        perm = dict(zip(up, rng.permutation(up)))          # shuffle residues, keep rows consistent
                        src = {p: Xte.loc[pte == q, bc].iloc[0].values for p, q in perm.items()}
                        Xp[bc] = np.vstack([src[p] for p in pte])
                    else:
                        Xp[bc] = Xte[bc].values[rng.permutation(len(Xte))]
                    drops[b].append(base - spearmanr(m.predict(Xp), yte)[0])
    for b, v in drops.items():
        importances.append(dict(assay=assay, feature_block=b, spearman_drop=round(float(np.mean(v)), 4)))
    print("  importance (drop in held-out Spearman when shuffled):",
          {b: round(float(np.mean(v)), 3) for b, v in drops.items()})

pd.DataFrame(results).to_csv(HERE / "results_models.csv", index=False)
pd.DataFrame(contrasts).to_csv(HERE / "results_contrasts.csv", index=False)
pd.DataFrame(importances).to_csv(HERE / "results_importance.csv", index=False)
print("\nsaved results")
