"""
AlphaMissense comparison: exactly the pre-specified analysis in PLAN.md.
Same protocol as ../variant_ml/run_variant_ml.py. Do not edit settings after seeing results.
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
esm = pd.read_csv(ROOT / "variant_ml" / "esm_llr.csv")

am = pd.read_csv(HERE / "AF-P04637-aa-substitutions.csv")
_m = am.protein_variant.str.extract(r"^([A-Z])(\d+)([A-Z])$")
am["wt"], am["position"], am["mut"] = _m[0], _m[1].astype(int), _m[2]
am = am[["position", "wt", "mut", "am_pathogenicity"]].rename(columns={"am_pathogenicity": "am_score"})

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
    n0 = len(v)
    v = v.merge(am, on=["position", "wt", "mut"], how="inner")
    assert len(v) == n0, "AlphaMissense coverage / wild-type mismatch"
    v["d_hydro"] = v.mut.map(KD) - v.wt.map(KD)
    v["d_volume"] = v.mut.map(VOL) - v.wt.map(VOL)
    v["d_charge"] = v.mut.map(lambda a: CHG.get(a, 0)) - v.wt.map(lambda a: CHG.get(a, 0))
    v["blosum62"] = [B62[w][m] for w, m in zip(v.wt, v.mut)]
    v["wt_hydro"] = v.wt.map(KD); v["mut_hydro"] = v.mut.map(KD)
    v["y"] = (rankdata(v.score) - 1) / (len(v) - 1)
    return v.reset_index(drop=True)

SUBST = ["esm_llr", "d_hydro", "d_volume", "d_charge", "blosum62", "wt_hydro", "mut_hydro"]
SETS = {"G0 AlphaMissense alone": ["am_score"],
        "G1 S + AlphaMissense": SUBST + ["am_score"],
        "G2 + burial": SUBST + ["am_score", "burial_rsa"],
        "G3 + burial + own value": SUBST + ["am_score", "burial_rsa", "own_value"],
        "G4 + burial + patch": SUBST + ["am_score", "burial_rsa", "patch_score"],
        "G5 + burial + own + patch": SUBST + ["am_score", "burial_rsa", "own_value", "patch_score"],
        # reference sets without AlphaMissense, refitted so paired bootstrap contrasts are possible
        "F2 (no AM) S + burial": SUBST + ["burial_rsa"],
        "F3 (no AM) S + burial + own value": SUBST + ["burial_rsa", "own_value"]}
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

prev = pd.read_csv(ROOT / "variant_ml" / "results_models.csv")
results, contrasts, importances = [], [], []
for assay, fname in (("Giacomelli 2018", "giacomelli_scores.csv"), ("Kato 2003", "kato_scores.csv")):
    d = build(fname)
    print(f"\n=== {assay}: {len(d)} substitutions at {d.position.nunique()} residues ===")
    print(f"  raw Spearman(AlphaMissense, damage) = {spearmanr(d.am_score, d.y)[0]:.3f}; "
          f"Spearman(ESM-2 score, damage) = {spearmanr(-d.esm_llr, d.y)[0]:.3f}")
    y = d.y.values; pos = d.position.values
    oof = {n: np.zeros((N_REP, len(d))) for n in SETS}
    for rep in range(N_REP):
        fid = fold_ids(pos, rep)
        for k in range(N_FOLD):
            tr, te = fid != k, fid == k
            for n, cols in SETS.items():
                m = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, cols], y[tr])
                oof[n][rep, te] = m.predict(d.loc[te, cols])
    pt = {}
    for n in SETS:
        rho = np.mean([spearmanr(oof[n][r], y)[0] for r in range(N_REP)])
        rr = np.mean([r2(y, oof[n][r]) for r in range(N_REP)])
        pt[n] = rho
        results.append(dict(assay=assay, features=n, oof_spearman=round(rho, 4), oof_R2=round(rr, 4),
                            n_rows=len(d), n_residues=d.position.nunique()))
        print(f"  {n:36s} OOF Spearman {rho:.3f}   R2 {rr:.3f}")
    # consistency check: refitted reference sets must reproduce the earlier variant-level result
    for n, old in (("F2 (no AM) S + burial", "F2 + burial"), ("F3 (no AM) S + burial + own value", "F3 + burial + own value")):
        o = prev[(prev.assay == assay) & (prev.features == old)].oof_spearman.iloc[0]
        print(f"  check {n}: now {pt[n]:.4f} vs earlier {o:.4f}  (diff {pt[n]-o:+.4f})")

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
    CONTR = (("PRIMARY 1: G5 vs G3 (patch added to S+AM+burial+own)", "G5 + burial + own + patch", "G3 + burial + own value"),
             ("PRIMARY 2: G4 vs G2 (patch added to S+AM+burial)", "G4 + burial + patch", "G2 + burial"),
             ("Secondary: G2 vs F2 (AlphaMissense added to S+burial)", "G2 + burial", "F2 (no AM) S + burial"),
             ("Secondary: G3 vs F3 (AlphaMissense added to S+burial+own)", "G3 + burial + own value", "F3 (no AM) S + burial + own value"))
    for label, a, b in CONTR:
        bd = boot_diff(a, b)
        contrasts.append(dict(assay=assay, contrast=label, delta_spearman=round(pt[a] - pt[b], 4),
                              ci_low=round(np.percentile(bd, 2.5), 4), ci_high=round(np.percentile(bd, 97.5), 4)))
        print(f"  {label}: delta {pt[a]-pt[b]:+.4f}  95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}]")

    cols = SETS["G5 + burial + own + patch"]
    blocks = {"AlphaMissense score": ["am_score"], "ESM-2 zero-shot score": ["esm_llr"],
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
                        perm = dict(zip(up, rng.permutation(up)))
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

# ---------------- residue level, stability data (descriptive) ----------------
print("\n=== Residue level: 37 residues with measured |ddG| ===")
exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")[["position", "abs_rep_ddg"]]
ala = pd.read_csv(ROOT / "results" / "ala_scan_cA.csv")[["position", "ddG"]].rename(columns={"ddG": "own"})
res = am.groupby("position", as_index=False).am_score.mean().rename(columns={"am_score": "am_mean"})
d37 = exp.merge(ala, on="position").merge(full[["position", "burial_rsa", "patch_score"]], on="position").merge(res, on="position")
assert len(d37) == 37
rows = []
for name, col, sign in (("AlphaMissense (residue mean)", "am_mean", 1), ("burial (1 - RSA)", "burial_rsa", -1),
                        ("own alanine-scan value", "own", 1), ("patch score", "patch_score", 1)):
    rho, p = spearmanr(sign * d37[col], d37.abs_rep_ddg)
    rows.append(dict(descriptor=name, spearman_with_abs_ddG=round(rho, 3), p=round(p, 4), n=37))
    print(f"  {name:30s} Spearman with |ddG| = {rho:+.3f} (p = {p:.4f})")
f194 = full.merge(res, on="position")
for name, col in (("burial (RSA)", "burial_rsa"), ("patch score", "patch_score")):
    rho = spearmanr(f194.am_mean, f194[col])[0]
    rows.append(dict(descriptor=f"AM residue mean vs {name} (194 residues)", spearman_with_abs_ddG=round(rho, 3), p=np.nan, n=len(f194)))
    print(f"  AlphaMissense residue mean vs {name} across {len(f194)} residues: Spearman {rho:+.3f}")
pd.DataFrame(rows).to_csv(HERE / "results_residue_level.csv", index=False)
print("\nsaved results")
