"""
Step 1 of the variant-level test (see PLAN): ESM-2 (150M) zero-shot substitution scores for TP53
positions 96-289. Masked-marginal log-likelihood ratio: mask the position in the FULL-LENGTH
sequence, then LLR(mut) = log p(mut) - log p(wt). No labels, no training.
The UniProt sequence is checked against the structure's sequence before use.
"""
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from Bio.PDB import PDBParser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
three1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
          "H1S":"H","H2S":"H","H3S":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
          "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

fa = urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/P04637.fasta").read().decode()
full = "".join(fa.splitlines()[1:])
print("UniProt P04637 length:", len(full))

chain = PDBParser(QUIET=True).get_structure("p", str(ROOT / "structure" / "tp53_chainA_zinc.pdb"))[0]["A"]
struct = {r.id[1]: three1.get(r.resname, "X") for r in chain if r.id[0] == " "}
bad = [p for p, a in struct.items() if full[p - 1] != a]
print(f"structure vs UniProt identity: {len(struct) - len(bad)}/{len(struct)} positions match", bad[:5])
assert not bad, "numbering mismatch"

name = "facebook/esm2_t30_150M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
model = EsmForMaskedLM.from_pretrained(name).eval()
aas = "ACDEFGHIKLMNPQRSTVWY"
aa_id = {a: tok.convert_tokens_to_ids(a) for a in aas}

ids = tok(full, return_tensors="pt")["input_ids"]
rows = []
with torch.no_grad():
    for p in sorted(struct):
        x = ids.clone()
        x[0, p] = tok.mask_token_id                   # index p = residue p (BOS at 0)
        lp = torch.log_softmax(model(input_ids=x).logits[0, p], dim=-1)
        wt = full[p - 1]
        for a in aas:
            if a != wt:
                rows.append((p, wt, a, float(lp[aa_id[a]] - lp[aa_id[wt]])))
        if p % 25 == 0:
            print("position", p, flush=True)
out = pd.DataFrame(rows, columns=["position", "wt", "mut", "esm_llr"])
out.to_csv(HERE / "esm_llr.csv", index=False)
print("saved", out.shape)
