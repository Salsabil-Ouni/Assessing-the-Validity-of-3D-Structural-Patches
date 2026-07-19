"""
Figures for the prof pipeline:
  1) results/figures/ala_scan_profile.png  - alanine-scan ddG along the sequence
  2) results/figures/patch_3D.png          - 3D view (CA atoms) of the hotspot patch
  3) results/figures/Q2_scatter.png        - |rep ddG| vs aggregated 3D-patch score
Also writes results/figures/patch_view.pml - a PyMOL script for a publication figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
HOT  = cfg["cutoffs"]["ala_hotspot_ddg"]
CHAIN = cfg["structure"]["chain"]
PDB  = ROOT / cfg["structure"]["repaired_pdb"]
FIG  = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

ala = pd.read_csv(ROOT / "results" / "ala_scan.csv")
rep = pd.read_csv(ROOT / "results" / "variant_microenvironment.csv")
patches = pd.read_csv(ROOT / "results" / "patches_3D.csv")
patch_members = set()
for _, r in patches.iterrows():
    if r["n_residues"] >= 2:
        patch_members |= {int(x) for x in str(r["residues"]).split(",")}

# ---------- 1) alanine-scan profile ----------------------------------------
fig, ax = plt.subplots(figsize=(13, 4))
colors = ["#d62728" if d >= HOT else "#9aa0a6" for d in ala["ddG"]]
ax.bar(ala["position"], ala["ddG"], color=colors, width=0.9)
ax.axhline(HOT, color="black", ls="--", lw=1, label=f"hotspot cutoff = {HOT}")
# mark experimentally destabilizing residues
dest = rep[rep["class"] == "destabilizing"]
ymax = ala["ddG"].max()
for _, r in dest.iterrows():
    ax.plot(r["position"], ymax + 0.4, marker="v", color="navy", ms=5)
ax.plot([], [], "v", color="navy", label="experimentally destabilizing residue")
ax.set_xlabel("residue position"); ax.set_ylabel("alanine-scan ddG (kcal/mol)")
ax.set_title("TP53 DBD alanine-scanning profile (chain A); red = hotspot")
ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
fig.savefig(FIG / "ala_scan_profile.png", dpi=150); plt.close(fig)

# ---------- 2) 3D structure view (CA atoms) --------------------------------
model = PDBParser(QUIET=True).get_structure("p", str(PDB))[0]
ca = {}
for res in model[CHAIN]:
    if "CA" in res:
        ca[res.id[1]] = res["CA"].coord
zinc = [a.coord for ch in model for r in ch for a in r
        if (a.element or "").strip().upper() == "ZN"]

pos = np.array(sorted(ca)); xyz = np.array([ca[p] for p in pos])
in_patch = np.array([p in patch_members for p in pos])
dest_pos = set(rep[rep["class"] == "destabilizing"]["position"])
is_dest  = np.array([p in dest_pos for p in pos])

fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
# backbone trace
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="lightgrey", lw=1, zorder=1)
ax.scatter(*xyz[~in_patch].T, c="#cfd2d6", s=18, label="non-patch residue")
ax.scatter(*xyz[in_patch].T, c="#ff9800", s=35, label="hotspot patch")
ax.scatter(*xyz[is_dest].T, facecolors="none", edgecolors="#d62728", s=90,
           lw=1.6, label="exp. destabilizing residue")
if zinc:
    z = np.array(zinc)[:1]   # chain A zinc
    ax.scatter(*z.T, c="black", marker="X", s=120, label="Zn")
ax.set_title("Hotspot 3D patch on TP53 DBD (chain A)")
ax.legend(loc="upper left", fontsize=8); ax.set_axis_off(); fig.tight_layout()
fig.savefig(FIG / "patch_3D.png", dpi=150); plt.close(fig)

# ---------- 3) Q2 scatter --------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
cmap = {"destabilizing": "#d62728", "neutral": "#9aa0a6", "stabilizing": "#1f77b4"}
for cl, g in rep.groupby("class"):
    ax.scatter(g["aggregated_patch_score"], g["abs_rep_ddg"],
               c=cmap[cl], label=cl, s=45, edgecolor="k", lw=0.3)
# trend line
z = np.polyfit(rep["aggregated_patch_score"], rep["abs_rep_ddg"], 1)
xs = np.linspace(rep["aggregated_patch_score"].min(), rep["aggregated_patch_score"].max(), 50)
ax.plot(xs, np.polyval(z, xs), "k--", lw=1, label="linear trend")
ax.set_xlabel("aggregated 3D-patch score (sum of neighbour alanine-scan ddG)")
ax.set_ylabel("|representative experimental ddG| (kcal/mol)")
ax.set_title("Q2: variant impact vs microenvironment score")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG / "Q2_scatter.png", dpi=150); plt.close(fig)

# ---------- PyMOL script ---------------------------------------------------
patch_sel = "+".join(map(str, sorted(patch_members)))
dest_sel  = "+".join(map(str, sorted(dest_pos)))
pml = f"""# PyMOL script - TP53 hotspot patch figure
# run:  pymol patch_view.pml      (after installing PyMOL)
load {PDB.as_posix()}, tp53
hide everything
bg_color white
create chA, tp53 and chain {CHAIN}
hide everything, tp53
show cartoon, chA
color grey80, chA
select patch, chA and resi {patch_sel}
color orange, patch
show sticks, patch and not name C+N+O
select destab, chA and resi {dest_sel}
color red, destab
show sticks, destab and not name C+N+O
show spheres, chA and elem Zn
color slate, chA and elem Zn
set cartoon_transparency, 0.2
orient chA
ray 1200, 1000
png patch_view.png, dpi=150
"""
(FIG / "patch_view.pml").write_text(pml, encoding="utf-8")

print("Saved figures to results/figures/:")
for f in ["ala_scan_profile.png", "patch_3D.png", "Q2_scatter.png", "patch_view.pml"]:
    print("  ", f)
