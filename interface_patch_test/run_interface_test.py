"""
Does spatial aggregation add value for an interface property burial cannot see?
Exactly the pre-specified analysis in PLAN.md (frozen). Same protocol as ../variant_ml/run_variant_ml.py.
Do not edit settings after seeing results; deviations must be logged in README.md.
"""
import itertools
import re
import sys
for _m in ("dask", "dask.array", "dask.dataframe", "dask.distributed"):
    sys.modules[_m] = None   # optional dask import in lightgbm breaks on this pandas; unused here
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from scipy.stats import spearmanr, rankdata
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FE = ROOT / "functional_ext" / "data"
rng = np.random.default_rng(0)
CLIP = 10.0            # A, fixed in PLAN.md
CONTACT = 5.0          # A, contact stratum (descriptive only)
N_PERM = 100

cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DIST = cfg["cutoffs"]["patch_distance_angstrom"]
CHAIN = cfg["structure"]["chain"]
PDB = ROOT / cfg["structure"]["repaired_pdb"]

# ------------------------- residue table, graph, interface property -------------------------
full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")[["position", "rsa", "patch_score", "ala_ddg"]]
ala_ddg = dict(zip(full.position, full.ala_ddg))
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg and r.id[0] == " "}
positions = sorted(heavy)
neigh = {p: set() for p in positions}
for p, q in itertools.combinations(positions, 2):
    if np.sqrt(((heavy[p][:, None, :] - heavy[q][None, :, :]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)
# check: this graph must reproduce the thesis's own patch scores (own value + neighbours' values)
recomputed = {p: ala_ddg[p] + sum(ala_ddg[q] for q in neigh[p]) for p in positions}
diff = max(abs(recomputed[p] - v) for p, v in zip(full.position, full.patch_score) if p in recomputed)
print(f"graph check: {len(positions)} residues, cut-off {DIST} A; max |recomputed - thesis patch score| = {diff:.2e}")
assert diff < 1e-3, "neighbour graph does not reproduce the thesis patch scores"

dna = pd.read_csv(ROOT / "results" / "dna_proximity.csv")
dna = dna[dna.position.isin(positions)]
assert set(dna.position) == set(positions), "DNA table and structure residues differ"
# numbering check for the DNA-bound structure: residue identities must match the wild types used elsewhere
esm = pd.read_csv(HERE.parent / "variant_ml" / "esm_llr.csv")
wt_of = esm.drop_duplicates("position").set_index("position").wt.to_dict()
t1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
      "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
m1tup = PDBParser(QUIET=True).get_structure("t", str(ROOT / "data" / "candidates" / "1TUP.pdb"))[0]
bad = checked = 0
for ch in ("A", "B", "C"):
    for r in m1tup[ch]:
        if r.id[0] == " " and r.id[1] in wt_of and r.resname in t1:
            checked += 1
            bad += t1[r.resname] != wt_of[r.id[1]]
print(f"1TUP numbering check: {checked - bad}/{checked} residue identities match")
assert checked > 300 and bad == 0, "1TUP numbering mismatch"

interface = dict(zip(dna.position, -np.minimum(dna.min_dist_dna.values, CLIP)))       # I_i = -min(d, 10)
def aggregate(I):
    return {p: sum(I[q] for q in neigh[p]) for p in positions}                         # A_i, excludes i itself
agg = aggregate(interface)
res = full.copy()
res["interface_own"] = res.position.map(interface)
res["interface_agg"] = res.position.map(agg)
res["min_dist_dna"] = res.position.map(dict(zip(dna.position, dna.min_dist_dna)))
print("context, 194 residues (Spearman): A vs I = %.3f | A vs burial(RSA) = %.3f | I vs burial(RSA) = %.3f | I vs packing patch = %.3f"
      % (spearmanr(res.interface_agg, res.interface_own)[0], spearmanr(res.interface_agg, res.rsa)[0],
         spearmanr(res.interface_own, res.rsa)[0], spearmanr(res.interface_own, res.patch_score)[0]))
pd.DataFrame([dict(pair="A vs I", rho=spearmanr(res.interface_agg, res.interface_own)[0]),
              dict(pair="A vs RSA", rho=spearmanr(res.interface_agg, res.rsa)[0]),
              dict(pair="I vs RSA", rho=spearmanr(res.interface_own, res.rsa)[0]),
              dict(pair="I vs packing patch", rho=spearmanr(res.interface_own, res.patch_score)[0])]
             ).round(3).to_csv(HERE / "interface_context.csv", index=False)

# ------------------------- functional data (identical to the variant-level test) -------------------------
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

am = pd.read_csv(HERE.parent / "alphamissense_test" / "AF-P04637-aa-substitutions.csv")
_m = am.protein_variant.str.extract(r"^([A-Z])(\d+)([A-Z])$")
am["wt"], am["position"], am["mut"] = _m[0], _m[1].astype(int), _m[2]
am = am[["position", "wt", "mut", "am_pathogenicity"]].rename(columns={"am_pathogenicity": "am_score"})
esm_llr = esm[["position", "wt", "mut", "esm_llr"]]

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
    v = (v.merge(esm_llr, on=["position", "wt", "mut"], how="inner")
          .merge(res[["position", "rsa", "patch_score", "ala_ddg", "interface_own", "interface_agg", "min_dist_dna"]], on="position", how="inner")
          .merge(am, on=["position", "wt", "mut"], how="inner"))
    v = v.rename(columns={"rsa": "burial_rsa", "ala_ddg": "own_value"})
    v["d_hydro"] = v.mut.map(KD) - v.wt.map(KD)
    v["d_volume"] = v.mut.map(VOL) - v.wt.map(VOL)
    v["d_charge"] = v.mut.map(lambda a: CHG.get(a, 0)) - v.wt.map(lambda a: CHG.get(a, 0))
    v["blosum62"] = [B62[w][m] for w, m in zip(v.wt, v.mut)]
    v["wt_hydro"] = v.wt.map(KD); v["mut_hydro"] = v.mut.map(KD)
    v["y"] = (rankdata(v.score) - 1) / (len(v) - 1)
    return v.reset_index(drop=True)

SUBST = ["esm_llr", "d_hydro", "d_volume", "d_charge", "blosum62", "wt_hydro", "mut_hydro"]
BASE = SUBST + ["burial_rsa", "own_value", "patch_score"]
SETS = {"J0 S+B+O+P (no interface term)": BASE,
        "J1 J0 + own interface value": BASE + ["interface_own"],
        "J2 J1 + aggregated interface score": BASE + ["interface_own", "interface_agg"],
        "J0a J0 + AlphaMissense": BASE + ["am_score"],
        "J1a J1 + AlphaMissense": BASE + ["interface_own", "am_score"],
        "J2a J2 + AlphaMissense": BASE + ["interface_own", "interface_agg", "am_score"]}
J1, J2 = "J1 J0 + own interface value", "J2 J1 + aggregated interface score"
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0, n_jobs=4, verbose=-1)
N_REP, N_FOLD, N_BOOT = 5, 5, 2000

def fold_ids(positions_, rep):
    r = np.random.default_rng(1000 + rep)
    up = r.permutation(np.unique(positions_))
    fid = {p: i % N_FOLD for i, p in enumerate(up)}
    return np.array([fid[p] for p in positions_])

def oof_predict(d, cols, pos):
    out = np.zeros((N_REP, len(d)))
    for rep in range(N_REP):
        fid = fold_ids(pos, rep)
        for k in range(N_FOLD):
            tr, te = fid != k, fid == k
            m = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, cols], d.y.values[tr])
            out[rep, te] = m.predict(d.loc[te, cols])
    return out

def mean_rho(pred, y, sel=None):
    if sel is None:
        return float(np.mean([spearmanr(pred[r], y)[0] for r in range(N_REP)]))
    return float(np.mean([spearmanr(pred[r][sel], y[sel])[0] for r in range(N_REP)]))

results, contrasts, perm_rows, stratum_rows = [], [], [], []
for assay, fname in (("Giacomelli 2018", "giacomelli_scores.csv"), ("Kato 2003", "kato_scores.csv")):
    d = build(fname)
    y = d.y.values; pos = d.position.values
    print(f"\n=== {assay}: {len(d)} substitutions at {d.position.nunique()} residues ===")
    oof = {n: oof_predict(d, cols, pos) for n, cols in SETS.items()}
    pt = {n: mean_rho(oof[n], y) for n in SETS}
    for n in SETS:
        r2 = np.mean([1 - ((y - oof[n][r]) ** 2).sum() / ((y - y.mean()) ** 2).sum() for r in range(N_REP)])
        results.append(dict(assay=assay, features=n, oof_spearman=round(pt[n], 4), oof_R2=round(float(r2), 4),
                            n_rows=len(d), n_residues=d.position.nunique()))
        print(f"  {n:38s} OOF Spearman {pt[n]:.4f}   R2 {r2:.4f}")

    idx_by_pos = {p: np.where(pos == p)[0] for p in np.unique(pos)}
    upos = np.array(list(idx_by_pos))
    def boot_diff(a, b):
        out = []
        for _ in range(N_BOOT):
            sel = np.concatenate([idx_by_pos[p] for p in rng.choice(upos, len(upos))])
            out.append(mean_rho(oof[a], y, sel) - mean_rho(oof[b], y, sel))
        return np.array(out)
    for label, a, b in (("PRIMARY: J2 vs J1 (aggregated interface score added)", J2, J1),
                        ("Secondary: J1 vs J0 (own interface value added)", J1, "J0 S+B+O+P (no interface term)"),
                        ("Secondary: J2a vs J1a (aggregated added, with AlphaMissense)", "J2a J2 + AlphaMissense", "J1a J1 + AlphaMissense"),
                        ("Secondary: J1a vs J0a (own interface added, with AlphaMissense)", "J1a J1 + AlphaMissense", "J0a J0 + AlphaMissense")):
        bd = boot_diff(a, b)
        contrasts.append(dict(assay=assay, contrast=label, delta_spearman=round(pt[a] - pt[b], 4),
                              ci_low=round(np.percentile(bd, 2.5), 4), ci_high=round(np.percentile(bd, 97.5), 4)))
        print(f"  {label}: delta {pt[a]-pt[b]:+.4f}  95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}]")

    # contact-only stratum, descriptive (no inference)
    sel = np.where(d.min_dist_dna.values <= CONTACT)[0]
    j1s, j2s = mean_rho(oof[J1], y, sel), mean_rho(oof[J2], y, sel)
    stratum_rows.append(dict(assay=assay, n_rows=len(sel), n_residues=int(d.position.iloc[sel].nunique()),
                             spearman_J1=round(j1s, 4), spearman_J2=round(j2s, 4), delta=round(j2s - j1s, 4)))
    print(f"  contact stratum (<= {CONTACT} A): {len(sel)} rows, {d.position.iloc[sel].nunique()} residues; "
          f"J1 {j1s:.4f} J2 {j2s:.4f} delta {j2s-j1s:+.4f} (descriptive)")

    # spatial-arrangement control: permute the interface property across residues, keep the graph
    obs = pt[J2] - pt[J1]
    prng = np.random.default_rng(12345)
    gains = []
    cols_j2 = SETS[J2]
    for k in range(N_PERM):
        vals = np.array([interface[p] for p in positions]); prng.shuffle(vals)
        perm_agg = aggregate(dict(zip(positions, vals)))
        dd = d.copy(); dd["interface_agg"] = dd.position.map(perm_agg)
        gains.append(mean_rho(oof_predict(dd, cols_j2, pos), y) - pt[J1])
        if (k + 1) % 20 == 0:
            print(f"    permutation {k+1}/{N_PERM}", flush=True)
    gains = np.array(gains)
    p_perm = (1 + np.sum(gains >= obs)) / (N_PERM + 1)
    perm_rows.append(dict(assay=assay, observed_gain=round(obs, 4), null_mean=round(gains.mean(), 4),
                          null_p95=round(np.percentile(gains, 95), 4), n_perm=N_PERM, p_upper=round(p_perm, 3)))
    print(f"  permutation control: observed gain {obs:+.4f}; null mean {gains.mean():+.4f}, 95th pct {np.percentile(gains,95):+.4f}; "
          f"share of permutations at least as large: {p_perm:.3f}")

pd.DataFrame(results).to_csv(HERE / "interface_models.csv", index=False)
pd.DataFrame(contrasts).to_csv(HERE / "interface_contrasts.csv", index=False)
pd.DataFrame(perm_rows).to_csv(HERE / "interface_permutation.csv", index=False)
pd.DataFrame(stratum_rows).to_csv(HERE / "interface_contact_stratum.csv", index=False)
print("\nsaved results")
