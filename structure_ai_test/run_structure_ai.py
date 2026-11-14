"""
Step 2 (PLAN.md, frozen): does a 3D-environment AI score (ProteinMPNN) predict TP53 variant damage, and beyond what?
Same statistical protocol as ../variant_ml/run_variant_ml.py. Do not edit settings after seeing results.
"""
import re
import sys
for _m in ("dask", "dask.array", "dask.dataframe", "dask.distributed"):
    sys.modules[_m] = None   # optional dask import in lightgbm breaks on this pandas; unused here
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, rankdata, mannwhitneyu
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from Bio.Align import substitution_matrices

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FE = ROOT / "functional_ext" / "data"
rng = np.random.default_rng(0)
PRIMARY_M = "2ocj_v_48_020_ctx"       # M_ctx, fixed in PLAN.md

# ------------------------------- inputs -------------------------------
sc_tab = pd.read_csv(HERE / "mpnn_scores.csv")
M_COLS = {"M_ctx": PRIMARY_M, "M_bb": "2ocj_v_48_020_bb", "M_ctx_ens": "2ocj_ens_ctx", "M_ctx_2xwr": "2xwr_v_48_020_ctx"}
for k, c in M_COLS.items():
    sc_tab[k] = -sc_tab[c]                      # damage direction: higher = more damaging
scores = sc_tab[["position", "wt", "mut"] + list(M_COLS)]

full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position", "rsa", "patch_score", "ala_ddg", "in_patch"]]
full = full.rename(columns={"rsa": "burial_rsa", "ala_ddg": "own_value"})
esm = pd.read_csv(ROOT / "variant_ml" / "esm_llr.csv")
am = pd.read_csv(ROOT / "alphamissense_test" / "AF-P04637-aa-substitutions.csv")
_x = am.protein_variant.str.extract(r"^([A-Z])(\d+)([A-Z])$")
am["wt"], am["position"], am["mut"] = _x[0], _x[1].astype(int), _x[2]
am = am[["position", "wt", "mut", "am_pathogenicity"]].rename(columns={"am_pathogenicity": "am_score"})

three1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I',
          'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
RX = re.compile(r'^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$')
HOTSPOTS = [('R',175,'H'),('R',248,'W'),('R',248,'Q'),('R',249,'S'),('G',245,'S'),('R',273,'H'),('R',282,'W')]
KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,L=3.8,K=-3.9,M=1.9,F=2.8,
          P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
VOL = dict(A=88.6,R=173.4,N=114.1,D=111.1,C=108.5,Q=143.8,E=138.4,G=60.1,H=153.2,I=166.7,L=166.7,K=168.6,
           M=162.9,F=189.9,P=112.7,S=89.0,T=116.1,W=227.8,Y=193.6,V=140.0)
CHG = dict(K=1, R=1, D=-1, E=-1)
B62 = substitution_matrices.load("BLOSUM62")

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

def add_chem(v):
    v["d_hydro"] = v.mut.map(KD) - v.wt.map(KD)
    v["d_volume"] = v.mut.map(VOL) - v.wt.map(VOL)
    v["d_charge"] = v.mut.map(lambda a: CHG.get(a, 0)) - v.wt.map(lambda a: CHG.get(a, 0))
    v["blosum62"] = [B62[w][m] for w, m in zip(v.wt, v.mut)]
    v["wt_hydro"] = v.wt.map(KD); v["mut_hydro"] = v.mut.map(KD)
    return v

def build(fname):
    v = orient(load(fname))
    v = v[(v.position >= 96) & (v.position <= 289)]
    v = v.groupby(["position", "wt", "mut"], as_index=False).score.mean()
    n0 = len(v)
    v = (v.merge(esm[["position", "wt", "mut", "esm_llr"]], on=["position", "wt", "mut"], how="inner")
          .merge(scores, on=["position", "wt", "mut"], how="inner")
          .merge(am, on=["position", "wt", "mut"], how="inner")
          .merge(full[["position", "burial_rsa", "patch_score", "own_value"]], on="position", how="inner"))
    assert len(v) == n0, "score coverage problem"
    v = add_chem(v)
    v["y"] = (rankdata(v.score) - 1) / (len(v) - 1)
    return v.reset_index(drop=True)

C = ["d_hydro", "d_volume", "d_charge", "blosum62", "wt_hydro", "mut_hydro"]
K0 = C + ["burial_rsa", "own_value", "patch_score"]
SETS = {"K0 classical (C+B+O+P)": K0,
        "K1 K0 + M_ctx": K0 + ["M_ctx"],
        "K2 K0 + ESM-2": K0 + ["esm_llr"],
        "K3 K2 + M_ctx": K0 + ["esm_llr", "M_ctx"],
        "K4 K2 + AlphaMissense": K0 + ["esm_llr", "am_score"],
        "K5 K4 + M_ctx": K0 + ["esm_llr", "am_score", "M_ctx"],
        "K6 K0 + M_bb": K0 + ["M_bb"],
        "K1e K0 + M_ctx (ensemble of noise levels)": K0 + ["M_ctx_ens"],
        "K1x K0 + M_ctx (2XWR structure)": K0 + ["M_ctx_2xwr"]}
N0 = "K0 classical (C+B+O+P)"
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0, n_jobs=4, verbose=-1)
N_REP, N_FOLD, N_BOOT = 5, 5, 2000

def fold_ids(positions_, rep):
    r = np.random.default_rng(1000 + rep)
    up = r.permutation(np.unique(positions_))
    fid = {p: i % N_FOLD for i, p in enumerate(up)}
    return np.array([fid[p] for p in positions_])

def mean_rho(pred, y, sel=None):
    if sel is None:
        return float(np.mean([spearmanr(pred[r], y)[0] for r in range(pred.shape[0])]))
    return float(np.mean([spearmanr(pred[r][sel], y[sel])[0] for r in range(pred.shape[0])]))

def raw_rho_ci(x, y, groups, n=2000):
    idx = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    ug = np.array(list(idx))
    pt = spearmanr(x, y)[0]
    bs = []
    for _ in range(n):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        bs.append(spearmanr(x[sel], y[sel])[0])
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

results, contrasts, raws = [], [], []
# ======================= 1. function (two assays) =======================
for assay, fname in (("Giacomelli 2018", "giacomelli_scores.csv"), ("Kato 2003", "kato_scores.csv")):
    d = build(fname)
    y = d.y.values; pos = d.position.values
    print(f"\n=== FUNCTION: {assay}: {len(d)} substitutions at {d.position.nunique()} residues ===")
    for lab, col, sgn in (("M_ctx (3D environment + neighbours)", "M_ctx", 1), ("M_bb (backbone geometry only)", "M_bb", 1),
                          ("ESM-2 (minus log ratio)", "esm_llr", -1), ("AlphaMissense", "am_score", 1)):
        pt, lo, hi = raw_rho_ci(sgn * d[col].values, y, pos)
        raws.append(dict(dataset=assay, score=lab, spearman=round(pt, 4), ci_low=round(lo, 4), ci_high=round(hi, 4)))
        print(f"  raw Spearman with damage, {lab:38s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    oof = {n: np.zeros((N_REP, len(d))) for n in SETS}
    for rep in range(N_REP):
        fid = fold_ids(pos, rep)
        for k in range(N_FOLD):
            tr, te = fid != k, fid == k
            for n, cols in SETS.items():
                m = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, cols], y[tr])
                oof[n][rep, te] = m.predict(d.loc[te, cols])
    pt = {n: mean_rho(oof[n], y) for n in SETS}
    for n in SETS:
        results.append(dict(dataset=assay, features=n, oof_spearman=round(pt[n], 4), n_rows=len(d), n_residues=d.position.nunique()))
        print(f"  {n:46s} OOF Spearman {pt[n]:.4f}")
    idx_by_pos = {p: np.where(pos == p)[0] for p in np.unique(pos)}
    upos = np.array(list(idx_by_pos))
    def boot_diff(a, b):
        out = []
        for _ in range(N_BOOT):
            sel = np.concatenate([idx_by_pos[p] for p in rng.choice(upos, len(upos))])
            out.append(mean_rho(oof[a], y, sel) - mean_rho(oof[b], y, sel))
        return np.array(out)
    for label, a, b in (("PRIMARY: K1 vs K0 (3D-environment AI added to classical features)", "K1 K0 + M_ctx", N0),
                        ("Secondary: K3 vs K2 (added beyond the language model)", "K3 K2 + M_ctx", "K2 K0 + ESM-2"),
                        ("Secondary: K5 vs K4 (added beyond language model + AlphaMissense)", "K5 K4 + M_ctx", "K4 K2 + AlphaMissense"),
                        ("Secondary: K1 vs K2 (structure AI compared with language AI)", "K1 K0 + M_ctx", "K2 K0 + ESM-2"),
                        ("Secondary: K6 vs K0 (backbone geometry only)", "K6 K0 + M_bb", N0),
                        ("Robustness: K1e vs K0 (ensemble of noise levels)", "K1e K0 + M_ctx (ensemble of noise levels)", N0),
                        ("Robustness: K1x vs K0 (second structure, 2XWR)", "K1x K0 + M_ctx (2XWR structure)", N0)):
        bd = boot_diff(a, b)
        contrasts.append(dict(dataset=assay, contrast=label, delta_spearman=round(pt[a] - pt[b], 4),
                              ci_low=round(np.percentile(bd, 2.5), 4), ci_high=round(np.percentile(bd, 97.5), 4)))
        print(f"  {label}: delta {pt[a]-pt[b]:+.4f}  95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}]")

# ======================= 2. stability (43 variants, 37 residues) =======================
print("\n=== STABILITY: 43 measured variants at 37 residues ===")
lv = pd.read_csv(ROOT / "results" / "listA_variants.csv")
sv = (lv.merge(scores, on=["position", "wt", "mut"], how="left").merge(esm[["position", "wt", "mut", "esm_llr"]], on=["position", "wt", "mut"], how="left")
        .merge(am, on=["position", "wt", "mut"], how="left").merge(full[["position", "burial_rsa", "patch_score", "own_value"]], on="position", how="left"))
assert len(sv) == 43 and sv[["M_ctx", "M_bb", "esm_llr", "am_score", "burial_rsa"]].notna().all().all(), "stability merge problem"
sv["esm_damage"] = -sv.esm_llr
y = (-sv.ddg).values                       # destabilisation (positive = destabilising)
yb = (sv.ddg < -1.0).astype(int).values
groups = sv.position.values
ALPHAS = np.logspace(-2, 3, 12)
def loo_ridge(X):
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        s = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(s.transform(X[tr]), y[tr])
        pred[te] = m.predict(s.transform(X[te]))
    return pred
R0c = ["burial_rsa", "own_value", "patch_score"]
SS = {"R0 burial + own + patch": R0c, "M_ctx alone": ["M_ctx"], "R1 R0 + M_ctx": R0c + ["M_ctx"],
      "R0 + ESM-2": R0c + ["esm_damage"], "R0 + AlphaMissense": R0c + ["am_score"],
      "R0 + ESM-2 + M_ctx": R0c + ["esm_damage", "M_ctx"], "R0 + AlphaMissense + M_ctx": R0c + ["am_score", "M_ctx"],
      "R0 + M_bb": R0c + ["M_bb"]}
spred = {n: loo_ridge(sv[c].values.astype(float)) for n, c in SS.items()}
stab_rows = []
for n, p in spred.items():
    rho = spearmanr(p, y)[0]; auc = roc_auc_score(yb, p)
    stab_rows.append(dict(features=n, LOO_spearman=round(rho, 3), LOO_AUC=round(auc, 3)))
    print(f"  {n:30s} LOO Spearman {rho:+.3f}  AUC {auc:.3f}")
ug = np.unique(groups); idxg = {g: np.where(groups == g)[0] for g in ug}
def sboot(a, b):
    out = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idxg[g] for g in rng.choice(ug, len(ug))])
        out.append(spearmanr(spred[a][sel], y[sel])[0] - spearmanr(spred[b][sel], y[sel])[0])
    return np.array(out)
scontr = []
for label, a, b in (("PRIMARY: R1 vs R0 (M_ctx added to burial + own + patch)", "R1 R0 + M_ctx", "R0 burial + own + patch"),
                    ("Secondary: (R0 + ESM-2 + M_ctx) vs (R0 + ESM-2)", "R0 + ESM-2 + M_ctx", "R0 + ESM-2"),
                    ("Secondary: (R0 + AlphaMissense + M_ctx) vs (R0 + AlphaMissense)", "R0 + AlphaMissense + M_ctx", "R0 + AlphaMissense"),
                    ("Secondary: R1 vs (R0 + ESM-2) (structure AI vs language AI)", "R1 R0 + M_ctx", "R0 + ESM-2"),
                    ("Secondary: (R0 + M_bb) vs R0 (backbone geometry only)", "R0 + M_bb", "R0 burial + own + patch")):
    bd = sboot(a, b); dl = spearmanr(spred[a], y)[0] - spearmanr(spred[b], y)[0]
    scontr.append(dict(contrast=label, delta_LOO_spearman=round(dl, 3), ci_low=round(np.percentile(bd, 2.5), 3), ci_high=round(np.percentile(bd, 97.5), 3)))
    print(f"  {label}: delta {dl:+.3f}  95% CI [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}]")
for lab, x in (("M_ctx", sv.M_ctx.values), ("M_bb", sv.M_bb.values), ("ESM-2 (minus log ratio)", sv.esm_damage.values),
               ("AlphaMissense", sv.am_score.values), ("burial (1 - RSA)", -sv.burial_rsa.values),
               ("own alanine-scan value", sv.own_value.values), ("patch score", sv.patch_score.values)):
    pt, lo, hi = raw_rho_ci(x, y, groups)
    raws.append(dict(dataset="Stability (43 variants)", score=lab, spearman=round(pt, 4), ci_low=round(lo, 4), ci_high=round(hi, 4)))
    print(f"  raw Spearman with destabilisation, {lab:26s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

# ======================= 3. residue level: does the AI agree with the hotspot patch? =======================
print("\n=== RESIDUE LEVEL (194 residues): AI sensitivity, computed without FoldX, alanine scanning or labels ===")
sens = scores.groupby("position").M_ctx.mean().rename("ai_sensitivity").reset_index()
res = full.merge(sens, on="position")
def partial_rho(x, yv, z):
    rx, ry, rz = rankdata(x), rankdata(yv), rankdata(z)
    def resid(a, b):
        A = np.vstack([np.ones_like(b), b]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])
rr = []
def rec(name, val, note=""):
    rr.append(dict(quantity=name, value=round(float(val), 3), note=note)); print(f"  {name:70s} {val:+.3f} {note}")
rec("Spearman(AI sensitivity, burial 1-RSA)", spearmanr(res.ai_sensitivity, -res.burial_rsa)[0])
rec("Spearman(AI sensitivity, packing patch score)", spearmanr(res.ai_sensitivity, res.patch_score)[0])
rec("  same, controlling for burial (partial Spearman)", partial_rho(res.ai_sensitivity, res.patch_score, res.burial_rsa))
inp = res[res.in_patch.astype(bool)].ai_sensitivity; outp = res[~res.in_patch.astype(bool)].ai_sensitivity
u = mannwhitneyu(inp, outp, alternative="two-sided")
rec("AUC: hotspot-patch residues more sensitive than others (AI sensitivity)", u.statistic / (len(inp) * len(outp)), f"(n in patch {len(inp)}, outside {len(outp)}; p={u.pvalue:.2g})")
rec("  same AUC using burial (1-RSA) alone", mannwhitneyu(-res[res.in_patch.astype(bool)].burial_rsa, -res[~res.in_patch.astype(bool)].burial_rsa)[0] / (len(inp) * len(outp)))
for assay, fname in (("Giacomelli 2018", "giacomelli_scores.csv"), ("Kato 2003", "kato_scores.csv")):
    v = orient(load(fname)); v = v[(v.position >= 96) & (v.position <= 289)]
    dm = v.groupby("position").score.mean().rename("damage").reset_index()
    z = res.merge(dm, on="position")
    rec(f"{assay}: Spearman(AI sensitivity, residue-mean damage)", spearmanr(z.ai_sensitivity, z.damage)[0], f"(n={len(z)})")
    rec(f"{assay}:   controlling for burial", partial_rho(z.ai_sensitivity, z.damage, z.burial_rsa))
    rec(f"{assay}: Spearman(burial 1-RSA, residue-mean damage), for comparison", spearmanr(-z.burial_rsa, z.damage)[0])
    rec(f"{assay}: Spearman(packing patch score, residue-mean damage), for comparison", spearmanr(z.patch_score, z.damage)[0])

pd.DataFrame(results).to_csv(HERE / "results_function_models.csv", index=False)
pd.DataFrame(contrasts).to_csv(HERE / "results_function_contrasts.csv", index=False)
pd.DataFrame(stab_rows).to_csv(HERE / "results_stability_models.csv", index=False)
pd.DataFrame(scontr).to_csv(HERE / "results_stability_contrasts.csv", index=False)
pd.DataFrame(raws).to_csv(HERE / "results_raw_correlations.csv", index=False)
pd.DataFrame(rr).to_csv(HERE / "results_residue_level.csv", index=False)
print("\nsaved results")
