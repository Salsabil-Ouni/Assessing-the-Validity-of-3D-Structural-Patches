"""
Clearer 3D figure of the hotspot patch on TP53 DBD (chain A + zinc, 2OCJ).
Fixes: (1) draws the protein BODY as a faint all-heavy-atom cloud so no point looks
'outside'; (2) explicit colours instead of rings so patch/destabilising OVERLAP is
obvious; (3) true geometry (box aspect) and two viewing angles.
"""
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DEL_BELOW, STAB_ABOVE = cfg["cutoffs"]["ddg_deleterious_below"], cfg["cutoffs"]["ddg_stabilising_above"]
PDB = ROOT / "structure" / "tp53_chainA_zinc.pdb"
FIG = ROOT / "results" / "figures"

rep = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")
patches = pd.read_csv(ROOT / "results" / "patches_3D_cA.csv")
patch_members = set()
for _, r in patches.iterrows():
    if r["n_residues"] >= 2:
        patch_members |= {int(x) for x in str(r["residues"]).split(",")}
dest_pos = set(rep.loc[rep["rep_ddg"] < DEL_BELOW, "position"])

model = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]
heavy_all = np.array([a.coord for r in model["A"] for a in r
                      if (a.element or "").strip().upper() not in ("H", "")])
ca = {r.id[1]: r["CA"].coord for r in model["A"] if "CA" in r}
zinc = np.array([a.coord for ch in model for r in ch for a in r
                 if (a.element or "").strip().upper() == "ZN"])

pos = sorted(ca); P = np.array([ca[p] for p in pos])
cat = []
for p in pos:
    inp, des = p in patch_members, p in dest_pos
    cat.append("both" if (inp and des) else "destab_out" if des else "patch" if inp else "other")
cat = np.array(cat)

n_both = int((cat == "both").sum()); n_out = int((cat == "destab_out").sum())

def draw(ax):
    # faint protein body (all heavy atoms)
    ax.scatter(*heavy_all.T, c="#d9dcde", s=2, alpha=0.05, linewidths=0)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#bfc3c7", lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(*P[cat == "other"].T, c="#c8ccd0", s=14, alpha=0.5, label="other residue")
    ax.scatter(*P[cat == "patch"].T, c="#ff9800", s=42, edgecolor="k", lw=0.3, label="hotspot patch")
    ax.scatter(*P[cat == "both"].T, c="#d62728", s=70, edgecolor="k", lw=0.4,
               label=f"destabilising IN patch (n={n_both})")
    ax.scatter(*P[cat == "destab_out"].T, c="#1f77b4", s=70, edgecolor="k", lw=0.4,
               label=f"destabilising OUTSIDE patch (n={n_out})")
    if len(zinc):
        ax.scatter(*zinc.T, c="black", marker="X", s=130, label="Zn")
    ax.set_box_aspect(np.ptp(P, axis=0)); ax.set_axis_off()

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection="3d"); draw(ax1); ax1.view_init(18, 30)
ax2 = fig.add_subplot(122, projection="3d"); draw(ax2); ax2.view_init(18, 150)
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=True)
fig.suptitle("Hotspot patch (orange = buried core) and experimentally destabilising residues\n"
             "TP53 DBD chain A + zinc (2OCJ); two views", fontsize=12)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "patch_3D.png", dpi=150); plt.close(fig)
print(f"saved patch_3D.png  | destabilising in patch={n_both}, outside={n_out}, "
      f"patch residues={int((cat=='patch').sum())+n_both}")
