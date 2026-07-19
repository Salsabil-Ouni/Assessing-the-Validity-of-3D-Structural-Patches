"""
Figures on the CONSISTENT-CONTEXT data (chain A + zinc, structure 2OCJ).
  1) ala_scan_profile.png   alanine-scan ddG along the sequence
  2) patch_3D.png           3D view of the hotspot patch (chain A + zinc)
  3) Q2_scatter.png         |experimental ddG| vs aggregated patch score
  4) validation_scatter.png FoldX-predicted (PositionScan) vs experimental ddG
  5) confounding_scatter.png  patch score vs RSA (n=194) - the burial confounder
Also rewrites patch_view.pml for the chain-A+zinc structure.
"""
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.stats import spearmanr
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
HOT = cfg["cutoffs"]["ala_hotspot_ddg"]
DEL_BELOW, STAB_ABOVE = cfg["cutoffs"]["ddg_deleterious_below"], cfg["cutoffs"]["ddg_stabilising_above"]
PDB = ROOT / "structure" / "tp53_chainA_zinc.pdb"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

ala  = pd.read_csv(ROOT / "results" / "ala_scan_cA.csv")
rep  = pd.read_csv(ROOT / "results" / "residue_table_cA.csv")          # 37 experimental
full = pd.read_csv(ROOT / "results" / "residue_table_full_cA.csv")     # 194 (+pred)
patches = pd.read_csv(ROOT / "results" / "patches_3D_cA.csv")
patch_members = set()
for _, r in patches.iterrows():
    if r["n_residues"] >= 2:
        patch_members |= {int(x) for x in str(r["residues"]).split(",")}

def classify(d):
    return "destabilizing" if d < DEL_BELOW else ("stabilizing" if d > STAB_ABOVE else "neutral")
rep["class"] = rep["rep_ddg"].apply(classify)
dest_pos = set(rep.loc[rep["class"] == "destabilizing", "position"])

# --- 1) alanine-scan profile ----------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(ala["position"], ala["ddG"],
       color=["#d62728" if d >= HOT else "#9aa0a6" for d in ala["ddG"]], width=0.9)
ax.axhline(HOT, color="black", ls="--", lw=1, label=f"hotspot cutoff = {HOT}")
ymax = ala["ddG"].max()
for p in dest_pos:
    ax.plot(p, ymax + 0.4, marker="v", color="navy", ms=5)
ax.plot([], [], "v", color="navy", label="experimentally destabilizing residue")
ax.set_xlabel("residue position"); ax.set_ylabel("alanine-scan ddG (kcal/mol)")
ax.set_title("TP53 DBD alanine-scan profile - chain A + zinc (2OCJ); red = hotspot")
ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
fig.savefig(FIG / "ala_scan_profile.png", dpi=150); plt.close(fig)

# --- 2) 3D patch view ------------------------------------------------------
model = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]
ca = {r.id[1]: r["CA"].coord for r in model["A"] if "CA" in r}
zinc = [a.coord for ch in model for r in ch for a in r if (a.element or "").strip().upper() == "ZN"]
pos = np.array(sorted(ca)); xyz = np.array([ca[p] for p in pos])
in_patch = np.array([p in patch_members for p in pos])
is_dest = np.array([p in dest_pos for p in pos])
fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="lightgrey", lw=1, zorder=1)
ax.scatter(*xyz[~in_patch].T, c="#cfd2d6", s=18, label="non-patch residue")
ax.scatter(*xyz[in_patch].T, c="#ff9800", s=35, label="hotspot patch")
ax.scatter(*xyz[is_dest].T, facecolors="none", edgecolors="#d62728", s=90, lw=1.6,
           label="exp. destabilizing residue")
if zinc:
    ax.scatter(*np.array(zinc).T, c="black", marker="X", s=120, label="Zn")
ax.set_title("Hotspot 3D patch on TP53 DBD (chain A + zinc, 2OCJ)")
ax.legend(loc="upper left", fontsize=8); ax.set_axis_off(); fig.tight_layout()
fig.savefig(FIG / "patch_3D.png", dpi=150); plt.close(fig)

# --- 3) Q2 scatter (experimental) -----------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
cmap = {"destabilizing": "#d62728", "neutral": "#9aa0a6", "stabilizing": "#1f77b4"}
for cl, g in rep.groupby("class"):
    ax.scatter(g["patch_score"], g["abs_rep_ddg"], c=cmap[cl], label=cl, s=45, edgecolor="k", lw=0.3)
z = np.polyfit(rep["patch_score"], rep["abs_rep_ddg"], 1)
xs = np.linspace(rep["patch_score"].min(), rep["patch_score"].max(), 50)
ax.plot(xs, np.polyval(z, xs), "k--", lw=1, label="linear trend")
rho, p = spearmanr(rep["abs_rep_ddg"], rep["patch_score"])
ax.set_xlabel("aggregated 3D-patch score"); ax.set_ylabel("|experimental ddG| (kcal/mol)")
ax.set_title(f"Q2 (experimental, n={len(rep)}): rho={rho:.2f}, p={p:.3g}")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG / "Q2_scatter.png", dpi=150); plt.close(fig)

# --- 4) validation: predicted vs experimental -----------------------------
val = full.merge(rep[["position", "rep_ddg", "class"]], on="position", how="inner")
fig, ax = plt.subplots(figsize=(6.5, 5))
for cl, g in val.groupby("class"):
    ax.scatter(g["pred_ddg_mean"], g["rep_ddg"], c=cmap[cl], label=cl, s=45, edgecolor="k", lw=0.3)
rho, p = spearmanr(val["pred_ddg_mean"], val["rep_ddg"])
ax.axhline(DEL_BELOW, color="grey", ls=":", lw=0.8)
ax.set_xlabel("FoldX-predicted mean ddG (PositionScan, +ve = destabilizing)")
ax.set_ylabel("experimental ddG (ThermoMutDB, -ve = destabilizing)")
ax.set_title(f"Validation (n={len(val)}): Spearman={rho:.2f}, p={p:.3g}")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG / "validation_scatter.png", dpi=150); plt.close(fig)

# --- 5) confounding: patch score vs RSA (n=194) ---------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(full.loc[~full.in_patch, "rsa"], full.loc[~full.in_patch, "patch_score"],
           c="#cfd2d6", s=25, label="non-patch", edgecolor="k", lw=0.2)
ax.scatter(full.loc[full.in_patch, "rsa"], full.loc[full.in_patch, "patch_score"],
           c="#ff9800", s=30, label="in patch", edgecolor="k", lw=0.2)
rho, p = spearmanr(full["patch_score"], full["rsa"])
ax.set_xlabel("relative solvent accessibility (RSA)"); ax.set_ylabel("aggregated 3D-patch score")
ax.set_title(f"Confounder: patch score vs RSA (n={len(full)}): Spearman={rho:.2f}")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG / "confounding_scatter.png", dpi=150); plt.close(fig)

# --- PyMOL script (chain A + zinc) ----------------------------------------
patch_sel = "+".join(map(str, sorted(patch_members)))
dest_sel = "+".join(map(str, sorted(dest_pos)))
(FIG / "patch_view.pml").write_text(f"""# PyMOL - TP53 hotspot patch (chain A + zinc, 2OCJ)
load {PDB.as_posix()}, tp53
hide everything
bg_color white
show cartoon
color grey80
select patch, resi {patch_sel}
color orange, patch
show sticks, patch and not name C+N+O
select destab, resi {dest_sel}
color red, destab
show sticks, destab and not name C+N+O
show spheres, elem Zn
color slate, elem Zn
orient
ray 1200, 1000
png patch_view.png, dpi=150
""", encoding="utf-8")

print("Regenerated figures (consistent context):")
for f in ["ala_scan_profile.png", "patch_3D.png", "Q2_scatter.png",
          "validation_scatter.png", "confounding_scatter.png", "patch_view.pml"]:
    print("  ", f)
