"""
Step 1 (PLAN.md): ProteinMPNN substitution scores for TP53, all 19 substitutions at every residue, computed before any
outcome is looked at. No retraining. Resumable: finished runs are skipped.
Score = log p(mutant) - log p(wild type), averaged over 5 runs (log-probabilities).
  ctx = conditioned on backbone AND the rest of the wild-type sequence;  bb = backbone geometry only.
"""
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MPNN = ROOT / "ai_pilot" / "ProteinMPNN"
OUT = HERE / "mpnn_raw"
OUT.mkdir(exist_ok=True)
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"      # ProteinMPNN residue alphabet
STRUCT = {"2ocj": (ROOT / "structure" / "tp53_chainA_zinc.pdb", "A", 96),
          "2xwr": (ROOT / "structure" / "tp53_2xwr_chainB_zinc.pdb", "B", 91)}
WEIGHTS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
MODES = {"ctx": ["--conditional_probs_only", "1"],
         "bb": ["--conditional_probs_only", "1", "--conditional_probs_only_backbone", "1"]}
N_RUNS = 5

esm = pd.read_csv(ROOT / "variant_ml" / "esm_llr.csv")
wt_of = esm.drop_duplicates("position").set_index("position").wt.to_dict()

todo = [(s, w, m) for s in STRUCT for w in WEIGHTS for m in MODES]
# primary first so that partial completion is useful
todo.sort(key=lambda t: (t[0] != "2ocj", t[1] != "v_48_020", t[2] != "ctx"))
for s, w, m in todo:
    pdb, chain, first = STRUCT[s]
    tag = f"{s}_{w}_{m}"
    npz = OUT / tag / "conditional_probs_only" / f"{pdb.stem}.npz"
    if npz.exists():
        print("have", tag, flush=True)
        continue
    t0 = time.time()
    cmd = [sys.executable, "protein_mpnn_run.py", "--pdb_path", pdb.as_posix(), "--pdb_path_chains", chain,
           "--out_folder", (OUT / tag).as_posix(), "--model_name", w,
           "--path_to_model_weights", (MPNN / "vanilla_model_weights").as_posix() + "/", "--seed", "37", "--batch_size", "1",
           "--num_seq_per_target", str(N_RUNS), "--backbone_noise", "0.0", "--suppress_print", "1"] + MODES[m]
    r = subprocess.run(cmd, cwd=str(MPNN), capture_output=True, text=True)
    if r.returncode != 0 or not npz.exists():
        print("FAILED", tag, r.stderr[-500:], flush=True)
        sys.exit(1)
    print(f"done {tag} in {time.time()-t0:.0f}s", flush=True)

# ---------------- assemble the score table ----------------
cols = {}
for s in STRUCT:
    pdb, chain, first = STRUCT[s]
    for w in WEIGHTS:
        for m in MODES:
            z = np.load(OUT / f"{s}_{w}_{m}" / "conditional_probs_only" / f"{pdb.stem}.npz")
            logp = z["log_p"].mean(axis=0)                         # [L, 21], mean over the 5 runs
            S = z["S"]
            seq = "".join(ALPHABET[i] for i in S)
            positions = list(range(first, first + len(seq)))
            bad = [p for p, a in zip(positions, seq) if p in wt_of and wt_of[p] != a]
            assert not bad, (s, "wild-type mismatch", bad[:5])
            rows = []
            for i, p in enumerate(positions):
                for a in "ACDEFGHIKLMNPQRSTVWY":
                    if a != seq[i]:
                        rows.append((p, seq[i], a, logp[i, ALPHABET.index(a)] - logp[i, ALPHABET.index(seq[i])]))
            cols[(s, w, m)] = pd.DataFrame(rows, columns=["position", "wt", "mut", "llr"]).set_index(["position", "wt", "mut"]).llr
tab = pd.concat({f"{s}_{w}_{m}": v for (s, w, m), v in cols.items()}, axis=1)
for s in STRUCT:
    for m in MODES:
        tab[f"{s}_ens_{m}"] = tab[[f"{s}_{w}_{m}" for w in WEIGHTS]].mean(axis=1)
tab = tab.reset_index()
tab = tab[tab.position.between(96, 289)]                            # the residues of the thesis analyses
tab.to_csv(HERE / "mpnn_scores.csv", index=False)
print("saved scores:", tab.shape, "| columns:", [c for c in tab.columns if c not in ("position", "wt", "mut")][:6], "...")
