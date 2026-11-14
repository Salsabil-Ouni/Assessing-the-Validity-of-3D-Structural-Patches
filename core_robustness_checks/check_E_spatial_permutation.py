"""Check E: structure-preserving permutation null for Q1, restricted to the 37 experimentally measured residues."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from Bio.PDB import PDBParser
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
N_PERM = 5000

cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DIST = cfg["cutoffs"]["patch_distance_angstrom"]
CHAIN = cfg["structure"]["chain"]
PDB = ROOT / cfg["structure"]["repaired_pdb"]

exp = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
assert len(exp) == 37
k_true = int(exp.in_patch.sum())
a_true = int((exp.in_patch & exp.destabilizing).sum())
print(f"n=37, k (true in-patch count)={k_true}, a (true in-patch AND destabilising)={a_true}, "
      f"destabilising total={int(exp.destabilizing.sum())}")

# adjacency among just these 37 residues, same 8A closest-heavy-atom rule as the main pipeline
chain = PDBParser(QUIET=True).get_structure("p", str(PDB))[0][CHAIN]
heavy = {r.id[1]: np.array([a.coord for a in r if (a.element or "").strip().upper() != "H"])
         for r in chain if r.id[1] in set(exp.position) and r.id[0] == " "}
positions = sorted(heavy)
assert positions == sorted(exp.position)
adj = {p: set() for p in positions}
for p, q in itertools.combinations(positions, 2):
    if np.sqrt(((heavy[p][:, None, :] - heavy[q][None, :, :]) ** 2).sum(-1)).min() <= DIST:
        adj[p].add(q); adj[q].add(p)
n_edges = sum(len(v) for v in adj.values()) // 2
print(f"adjacency among the 37 (<= {DIST} A): {n_edges} edges, "
      f"{sum(1 for p in positions if adj[p])} of 37 residues have at least one neighbour among the 37")

destab = dict(zip(exp.position, exp.destabilizing.astype(bool)))

def random_connected_subgraph(k, max_tries=200):
    for _ in range(max_tries):
        seed = positions[rng.integers(0, len(positions))]
        comp = {seed}
        frontier = list(adj[seed])
        while len(comp) < k and frontier:
            nxt = frontier.pop(rng.integers(0, len(frontier)))
            if nxt not in comp:
                comp.add(nxt)
                frontier.extend(x for x in adj[nxt] if x not in comp)
        if len(comp) == k:
            return comp
    return None

null_a = []
failures = 0
for _ in range(N_PERM):
    comp = random_connected_subgraph(k_true)
    if comp is None:
        failures += 1
        continue
    null_a.append(sum(destab[p] for p in comp))
null_a = np.array(null_a)
print(f"successful draws: {len(null_a)} / {N_PERM} ({failures} failed to reach size {k_true} in {200} tries "
      f"and were skipped)")

p_perm = (np.sum(null_a >= a_true) + 1) / (len(null_a) + 1)
print(f"\nobserved a={a_true} of k={k_true} in-patch residues destabilising")
print(f"null distribution over {len(null_a)} random size-{k_true} CONNECTED subgraphs of the 37-residue graph: "
      f"mean={null_a.mean():.2f}, 95th pct={np.percentile(null_a, 95):.1f}, max={null_a.max()}")
print(f"one-sided empirical p-value (P[null a >= observed a]) = {p_perm:.4f}")

# nominal Fisher's exact test on the same 2x2 table, for comparison
b_true = int((exp.in_patch & ~exp.destabilizing).sum())
c_true = int((~exp.in_patch & exp.destabilizing).sum())
d_true = int((~exp.in_patch & ~exp.destabilizing).sum())
odds, p_fisher = fisher_exact([[a_true, b_true], [c_true, d_true]])
print(f"\nfor reference, nominal Fisher's exact test on the same table: OR={odds:.2f}, p={p_fisher:.4f}")

pd.DataFrame({"null_a": null_a}).to_csv(HERE / "resultsE_permutation_null.csv", index=False)
pd.DataFrame([dict(k=k_true, a_observed=a_true, n_draws=len(null_a), null_mean=round(float(null_a.mean()), 3),
                   null_p95=float(np.percentile(null_a, 95)), p_permutation=round(p_perm, 4),
                   fisher_OR=round(odds, 3), fisher_p=round(p_fisher, 4))]).to_csv(
    HERE / "resultsE_summary.csv", index=False)
print("\nsaved resultsE_permutation_null.csv, resultsE_summary.csv")
