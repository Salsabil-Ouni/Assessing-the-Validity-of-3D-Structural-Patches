"""
AlphaMissense on the stability outcome: exactly the pre-specified analysis in PLAN_stability.md.
Protocol follows ../ai_pilot/evaluate_embeddings.py. Do not edit settings after seeing results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
rng = np.random.default_rng(0)

am = pd.read_csv(HERE / "AF-P04637-aa-substitutions.csv")
_m = am.protein_variant.str.extract(r"^([A-Z])(\d+)([A-Z])$")
am["wt"], am["position"], am["mut"] = _m[0], _m[1].astype(int), _m[2]
res_mean = am.groupby("position", as_index=False).am_pathogenicity.mean().rename(columns={"am_pathogenicity": "am_mean"})

exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
ala = pd.read_csv(ROOT / "results" / "ala_scan_cA.csv")[["position", "ddG"]].rename(columns={"ddG": "own"})
d = exp.merge(ala, on="position").merge(res_mean, on="position")
assert len(d) == 37

# substitution-matched AlphaMissense: the representative = most negative measured ddG at the residue
lv = pd.read_csv(ROOT / "results" / "listA_variants.csv")
rep = lv.loc[lv.groupby("position").ddg.idxmin()][["position", "wt", "mut", "ddg"]]
chk = d.merge(rep, on="position")
assert len(chk) == 37 and np.allclose(chk.rep_ddg, chk.ddg), "representative variant does not match residue table"
rep = rep.merge(am[["position", "wt", "mut", "am_pathogenicity"]], on=["position", "wt", "mut"], how="left")
assert rep.am_pathogenicity.notna().all(), "AlphaMissense missing for a measured variant"
d = d.merge(rep[["position", "am_pathogenicity"]].rename(columns={"am_pathogenicity": "am_var"}), on="position")

y = d.abs_rep_ddg.values
yb = d.destabilizing.astype(int).values
print(f"n = {len(d)} residues, {yb.sum()} destabilising")
print(f"raw Spearman with |ddG|: AM residue mean {spearmanr(d.am_mean, y)[0]:+.3f}, "
      f"AM of measured substitution {spearmanr(d.am_var, y)[0]:+.3f}, RSA(1-) {spearmanr(-d.rsa, y)[0]:+.3f}, "
      f"patch {spearmanr(d.patch_score, y)[0]:+.3f}, own {spearmanr(d.own, y)[0]:+.3f}")

ALPHAS = np.logspace(-2, 3, 12)
def loo_reg(X, y):
    n = len(y); pred = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        pred[i] = m.predict(sc.transform(X[i:i+1]))[0]
    return pred
def loo_clf(X, yb):
    n = len(yb); pred = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.3, max_iter=1000).fit(sc.transform(X[tr]), yb[tr])
        pred[i] = m.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    return pred
def perm_p(pred, y, obs, n=1000):
    null = np.array([spearmanr(pred, rng.permutation(y))[0] for _ in range(n)])
    return (np.sum(null >= obs) + 1) / (n + 1)

R, Pt, O = d[["rsa"]].values, d[["patch_score"]].values, d[["own"]].values
def sets(A):
    return {"AM": A, "RSA": R, "patch score": Pt, "own value": O,
            "RSA + patch": np.hstack([R, Pt]),
            "RSA + AM": np.hstack([R, A]), "patch + AM": np.hstack([Pt, A]),
            "RSA + patch + AM": np.hstack([R, Pt, A]),
            "own + AM": np.hstack([O, A]), "own + patch + AM": np.hstack([O, Pt, A])}

N_BOOT = 2000
rows, contrasts = [], []
for variant, col in (("residue-mean AM (primary)", "am_mean"), ("substitution-matched AM (secondary)", "am_var")):
    A = d[[col]].values
    preds = {}
    print(f"\n=== {variant} ===")
    for name, X in sets(A).items():
        pr = loo_reg(X, y); preds[name] = pr
        rho = spearmanr(pr, y)[0]
        r2 = 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        auc = roc_auc_score(yb, loo_clf(X, yb))
        p = perm_p(pr, y, rho) if name in ("AM", "RSA", "patch score", "own value") else np.nan
        rows.append(dict(descriptor_version=variant, features=name, LOO_spearman=round(rho, 3),
                         LOO_R2=round(r2, 3), perm_p=(round(p, 3) if not np.isnan(p) else ""), LOO_AUC=round(auc, 3)))
        print(f"  {name:20s} LOO Spearman {rho:+.3f}  R2 {r2:+.3f}  AUC {auc:.3f}" + (f"  perm p {p:.3f}" if not np.isnan(p) else ""))
    def boot(a, b):
        out = []
        for _ in range(N_BOOT):
            s = rng.integers(0, len(y), len(y))
            out.append(spearmanr(preds[a][s], y[s])[0] - spearmanr(preds[b][s], y[s])[0])
        return np.array(out)
    for label, a, b in (("PRIMARY 1: (AM + patch) vs AM", "patch + AM", "AM"),
                        ("PRIMARY 2: (RSA + AM + patch) vs (RSA + AM)", "RSA + patch + AM", "RSA + AM"),
                        ("Secondary: (RSA + AM) vs RSA", "RSA + AM", "RSA"),
                        ("Secondary: (patch + AM) vs patch", "patch + AM", "patch score"),
                        ("Secondary: (own + patch + AM) vs (own + AM)", "own + patch + AM", "own + AM")):
        bd = boot(a, b)
        dl = spearmanr(preds[a], y)[0] - spearmanr(preds[b], y)[0]
        contrasts.append(dict(descriptor_version=variant, contrast=label, delta_LOO_spearman=round(dl, 3),
                              ci_low=round(np.percentile(bd, 2.5), 3), ci_high=round(np.percentile(bd, 97.5), 3)))
        print(f"  {label}: delta {dl:+.3f}  95% CI [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}]")

pd.DataFrame(rows).to_csv(HERE / "stability_models.csv", index=False)
pd.DataFrame(contrasts).to_csv(HERE / "stability_contrasts.csv", index=False)
print("\nsaved results")
