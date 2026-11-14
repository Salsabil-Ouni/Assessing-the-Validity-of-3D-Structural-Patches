"""
Representative-value sensitivity check (directly answers the supervisor's own
whiteboard note on the variant-prioritisation step: "expect to have some
redundant variant measures - what should we do later statistically").

The main pipeline collapses multiple substitutions at one residue to a single
representative value using the MOST DESTABILISING measurement (min signed ddG).
Supporting docs (README.md, SUPERVISOR_SUMMARY.md) have long claimed the MEAN
representative gives the same conclusion "as a sensitivity check", but no
script or thesis chapter has actually reported that comparison with numbers.
This does so, on the same chain-A-consistent structural columns
(patch_score, rsa, in_patch) already computed for the main analysis - no new
structure or metric, purely a different way of collapsing ThermoMutDB
replicate measurements to the residue level.
"""
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from scipy.stats.contingency import odds_ratio
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]

A = pd.read_csv(RES / "listA_variants.csv")
struct = pd.read_csv(RES / "residue_table_cA.csv")[["position", "patch_score", "rsa", "in_patch"]]

n_multi = (A.groupby("position").size() > 1).sum()
print(f"Residues with >1 measured substitution: {n_multi} of {A.position.nunique()}")
print(A.groupby("position").agg(n=("ddg", "size"), ddgs=("ddg", list)).query("n > 1"))

def build(representative_col_fn, label):
    rep = A.groupby("position", as_index=False).agg(rep_ddg=("ddg", representative_col_fn))
    rep["destabilizing"] = rep["rep_ddg"] < DEL_BELOW
    rep["abs_rep_ddg"] = rep["rep_ddg"].abs()
    rep = rep.merge(struct, on="position", how="inner")

    a = int((rep.destabilizing & rep.in_patch).sum()); b = int((rep.destabilizing & ~rep.in_patch).sum())
    c = int((~rep.destabilizing & rep.in_patch).sum()); d = int((~rep.destabilizing & ~rep.in_patch).sum())
    o = odds_ratio([[a, b], [c, d]]); ci = o.confidence_interval(0.95)
    _, pf = fisher_exact([[a, b], [c, d]])
    rho, prho = spearmanr(rep.abs_rep_ddg, rep.patch_score)

    print(f"\n=== Representative = {label} (n={len(rep)}, destabilising={int(rep.destabilizing.sum())}) ===")
    print(f"  Q1 Fisher: table=({a},{b},{c},{d}) OR={o.statistic:.2f} CI[{ci.low:.2f},{ci.high:.2f}] p={pf:.4g}")
    print(f"  Q2 Spearman(|ddG|, patch_score): rho={rho:.3f} p={prho:.4g}")
    return {"representative": label, "n": len(rep), "n_destabilising": int(rep.destabilizing.sum()),
            "Q1_OR": round(o.statistic, 2), "Q1_CI_low": round(ci.low, 2), "Q1_CI_high": round(ci.high, 2),
            "Q1_p": round(pf, 5), "Q2_rho": round(rho, 3), "Q2_p": round(prho, 5)}

rows = [build("min", "most destabilising (main text)"), build("mean", "mean of replicates")]
pd.DataFrame(rows).to_csv(RES / "representative_sensitivity.csv", index=False)
print("\nSaved -> results/representative_sensitivity.csv")
