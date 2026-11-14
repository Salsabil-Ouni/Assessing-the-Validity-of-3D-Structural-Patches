"""
Sensitivity of the conclusion to the HOTSPOT threshold itself
(g_i >= ala_hotspot_ddg, config.yaml). This is the one parameter the earlier
robustness checks (sensitivity_cutoff.py: ddG cutoff; distance_sweep.py: 8A
neighbour radius) never varied - it decides which residues are even eligible
to form a patch.

Re-derives, at each hotspot threshold in HOT_SWEEP:
  - number of hotspots and how they group into patches (single core vs several)
  - Q1 (Fisher OR, 95% CI, p) on the full experimental set (n=37)
  - Q1 restricted to buried residues only (RSA < BURIED_REL) - the "beyond burial" test
  - RSA-adjusted logistic patch OR (in_patch + rsa -> destabilising)

Q2 (Spearman of |ddG| vs the continuous aggregated microenvironment score) is
NOT re-derived here: that score sums neighbour ala-scan ddG within the fixed
8A radius for ALL residues regardless of hotspot status, so it is unaffected by
where the hotspot threshold is drawn (already noted in consistent_context.py).
Only the binary hotspot/patch membership used by Q1 depends on this threshold.

Reuses: ala_scan_cA.csv (chain-A alanine scan), residue_table_full_cA.csv (rsa,
patch_score per position - both threshold-independent), listA_variants.csv
(experimental residue-level data), and the same chain-A + zinc geometry as
consistent_context.py.
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import yaml
import statsmodels.api as sm
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DIST = cfg["cutoffs"]["patch_distance_angstrom"]
DEL_BELOW = cfg["cutoffs"]["ddg_deleterious_below"]
BURIED_REL = 0.10  # matches buried_only.py's RSA_CUT (the "beyond burial" test used in results.tex),
                    # not config.yaml's more permissive buried_cutoff_relative=0.20
PDB = ROOT / "structure" / "tp53_chainA_zinc.pdb"
CURRENT_HOT = cfg["cutoffs"]["ala_hotspot_ddg"]

HOT_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0]

# ---- inputs shared across all thresholds (independent of the hotspot cutoff) ----
ala = pd.read_csv(RES / "ala_scan_cA.csv")
ala_ddg = dict(zip(ala.position, ala.ddG))

chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]["A"]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in ala_ddg}
positions = sorted(heavy)
neigh = {p: {p} for p in positions}
for p, q in itertools.combinations(positions, 2):
    if np.sqrt(((heavy[p][:, None] - heavy[q][None]) ** 2).sum(-1)).min() <= DIST:
        neigh[p].add(q); neigh[q].add(p)

full = pd.read_csv(RES / "residue_table_full_cA.csv")  # position, rsa, patch_score (threshold-independent)
rsa_of = dict(zip(full.position, full.rsa))

A = pd.read_csv(RES / "listA_variants.csv")
rep = (A.groupby("position", as_index=False)
         .agg(rep_ddg=("ddg", "min")))
rep["destabilizing"] = rep["rep_ddg"] < DEL_BELOW
rep["rsa"] = rep.position.map(rsa_of)
rep = rep.dropna(subset=["rsa"])


def fisher(df, in_patch_col="in_patch"):
    a = int((df.destabilizing & df[in_patch_col]).sum())
    b = int((df.destabilizing & ~df[in_patch_col]).sum())
    c = int((~df.destabilizing & df[in_patch_col]).sum())
    d = int((~df.destabilizing & ~df[in_patch_col]).sum())
    if min(a + b, c + d, a + c, b + d) == 0:
        return (a, b, c, d), None, None, None
    o = odds_ratio([[a, b], [c, d]]); ci = o.confidence_interval(0.95)
    _, p = fisher_exact([[a, b], [c, d]])
    stat = o.statistic if np.isfinite(o.statistic) else None
    return (a, b, c, d), stat, (ci.low, ci.high), p


rows = []
for hot in HOT_SWEEP:
    is_hot = {p: ala_ddg[p] >= hot for p in positions}
    hp = [p for p in positions if is_hot[p]]
    G = nx.Graph(); G.add_nodes_from(hp)
    for p, q in itertools.combinations(hp, 2):
        if q in neigh[p]:
            G.add_edge(p, q)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    patch_members = set().union(*[c for c in comps if len(c) >= 2]) if comps else set()
    n_real_patches = sum(1 for c in comps if len(c) >= 2)

    rep_h = rep.copy()
    rep_h["in_patch"] = rep_h.position.isin(patch_members)
    tab, orr, ci, pf = fisher(rep_h)

    buried = rep_h[rep_h.rsa < BURIED_REL]
    tab_b, orr_b, ci_b, pf_b = fisher(buried)

    y = rep_h.destabilizing.astype(int).values
    l2 = sm.Logit(y, sm.add_constant(rep_h[["in_patch", "rsa"]].astype(float))).fit(disp=0)
    adj_or, adj_p = float(np.exp(l2.params["in_patch"])), float(l2.pvalues["in_patch"])

    rows.append({
        "hotspot_cutoff": hot,
        "n_hotspots": len(hp),
        "n_patches(>=2)": n_real_patches,
        "largest_patch_size": max((len(c) for c in comps), default=0),
        "n_in_patch(exp set)": int(rep_h.in_patch.sum()),
        "Q1_OR": None if orr is None else round(orr, 2),
        "Q1_CI_low": None if orr is None else round(ci[0], 2),
        "Q1_CI_high": None if orr is None else round(ci[1], 2),
        "Q1_p": None if pf is None else round(pf, 4),
        "buried_only_OR": None if orr_b is None else round(orr_b, 2),
        "buried_only_p": None if pf_b is None else round(pf_b, 4),
        "RSA_adj_OR": round(adj_or, 2),
        "RSA_adj_p": round(adj_p, 4),
    })

out = pd.DataFrame(rows)
out.to_csv(RES / "hotspot_threshold_sweep.csv", index=False)

print(f"Neighbour radius fixed at {DIST} A; destabilising cutoff fixed at ddG < {DEL_BELOW}; "
      f"current pipeline hotspot cutoff = {CURRENT_HOT}\n")
print(out.to_string(index=False))

md = ["# Hotspot-threshold sensitivity sweep",
      "",
      f"Neighbour radius fixed at {DIST} Å; destabilising cutoff fixed at ΔΔG < "
      f"{DEL_BELOW} kcal/mol (both pre-registered elsewhere). Only the alanine-scan hotspot "
      f"threshold (g_i, config `ala_hotspot_ddg`, currently {CURRENT_HOT}) is varied here.",
      "",
      "| cutoff | hotspots | patches (n>=2) | largest patch | n in-patch (exp.) "
      "| Q1 OR (95% CI) p | buried-only OR (p) | RSA-adj. OR (p) |",
      "|---:|---:|---:|---:|---:|---|---|---|"]
for r in rows:
    q1 = "n/a" if r["Q1_OR"] is None else f"{r['Q1_OR']} [{r['Q1_CI_low']},{r['Q1_CI_high']}] p={r['Q1_p']}"
    bur = ("undefined (0 cell) "
           f"p={r['buried_only_p']}") if (r["buried_only_OR"] is None or pd.isna(r["buried_only_OR"])) \
        else f"{r['buried_only_OR']} (p={r['buried_only_p']})"
    md.append(f"| {r['hotspot_cutoff']} | {r['n_hotspots']} | {r['n_patches(>=2)']} | "
              f"{r['largest_patch_size']} | {r['n_in_patch(exp set)']} | {q1} | {bur} | "
              f"{r['RSA_adj_OR']} (p={r['RSA_adj_p']}) |")
md += ["",
       "Q2 (Spearman of |ΔΔG| vs. the continuous aggregated microenvironment score) is "
       "not re-derived here: that score sums neighbour alanine-scan ΔΔG within the fixed "
       f"{DIST} Å radius for every residue regardless of hotspot status, so it does not depend "
       "on where the hotspot threshold is drawn. Only the binary hotspot/patch membership used "
       "by Q1 (and by the RSA-adjusted logistic model) is threshold-dependent.",
       "",
       "**Reading the two loosest rows (1.0, 1.5):** the RSA-adjusted OR looks significant there, "
       "but at those cutoffs 84-98 of the 194 chain-A residues are called \"hotspots\" (roughly "
       "half the domain), so \"in patch\" has drifted from meaning a discrete, mechanistically "
       "distinct microenvironment to meaning little more than \"structured region of the protein\". "
       "The buried-only Fisher test is undefined at these two cutoffs because every buried "
       "experimental residue falls inside the (now enormous) patch, leaving an empty comparison "
       "cell - itself a symptom of the threshold being too loose to be informative, not evidence "
       "for an effect beyond burial."]
(RES / "hotspot_threshold_sweep.md").write_text("\n".join(md), encoding="utf-8")
print("\nSaved -> results/hotspot_threshold_sweep.csv, results/hotspot_threshold_sweep.md")
