"""Figure for the AlphaMissense comparison (white theme, matching the thesis's own figures), read
directly from the saved result tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SURF, INK, INK2, GRID = "#ffffff", "#1a1a19", "#444440", "#cfcec6"
BLUE, ORANGE = "#1f4e8c", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
mod = pd.read_csv(HERE / "results_models.csv")
con = pd.read_csv(HERE / "results_contrasts.csv")
ASSAYS = [("Giacomelli 2018", BLUE), ("Kato 2003", ORANGE)]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), dpi=170,
                         gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0], wspace=0.8))

def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

# A: accuracy by feature set
ax = axes[0]
LAB = {"G0 AlphaMissense alone": "G0 AlphaMissense alone",
       "G1 S + AlphaMissense": "G1 + substitution features",
       "G2 + burial": "G2: G1 + burial",
       "G3 + burial + own value": "G3: G2 + own value",
       "G4 + burial + patch": "G4: G2 + patch",
       "G5 + burial + own + patch": "G5: G3 + patch",
       "F2 (no AM) S + burial": "ref: no AlphaMissense,\nsubstitution + burial",
       "F3 (no AM) S + burial + own value": "ref: no AlphaMissense,\n+ own value"}
sets = list(mod.features.unique())[::-1]
y = np.arange(len(sets))
for j, (a, c) in enumerate(ASSAYS):
    v = [mod[(mod.assay == a) & (mod.features == s)].oof_spearman.iloc[0] for s in sets]
    ax.barh(y + (j - 0.5) * 0.36, v, height=0.32, color=c, edgecolor=SURF, linewidth=1.5)
    for yy, vv in zip(y + (j - 0.5) * 0.36, v):
        ax.text(vv + 0.005, yy, f"{vv:.3f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([LAB[s] for s in sets], fontsize=9)
ax.set_xlim(0.5, 0.79)
ax.set_xlabel("Held-out Spearman with measured damage", fontsize=9.5)
ax.set_title("A. AlphaMissense is the strongest input", loc="left", fontsize=11.5, fontweight="bold")
clean(ax)
ax.legend(handles=[Patch(color=c, label=a) for a, c in ASSAYS], loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2)

def forest(ax, prefixes, labels, xlim, title, xlabel):
    rows = []
    for a, c in ASSAYS:
        for pre, lab in zip(prefixes, labels):
            r = con[(con.assay == a) & con.contrast.str.startswith(pre)].iloc[0]
            rows.append((f"{a.split()[0]}\n{lab}", r.delta_spearman, r.ci_low, r.ci_high, c))
    rows = rows[::-1]
    for i, (lab, d, lo, hi, c) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round")
        ax.scatter([d], [i], s=90, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.text(d, i + 0.2, f"{d:+.3f}", ha="center", fontsize=9, color=INK2)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    clean(ax)

forest(axes[1], ["PRIMARY 2", "PRIMARY 1"],
       ["patch on top of\nAlphaMissense + burial", "patch on top of\nAlphaMissense + burial + own"],
       (-0.04, 0.04), "B. Patch score: every interval includes zero",
       "Gain in held-out Spearman from adding the patch score\n(dot = estimate, bar = 95% interval over residues)")
forest(axes[2], ["Secondary: G2", "Secondary: G3"],
       ["AlphaMissense on top of\nsubstitution + burial", "AlphaMissense on top of\nsubstitution + burial + own"],
       (-0.02, 0.135), "C. AlphaMissense: no interval includes zero",
       "Gain in held-out Spearman from adding AlphaMissense\n(dot = estimate, bar = 95% interval over residues)")

fig.suptitle("AlphaMissense comparison: a strong predictor helps, but the patch score still adds nothing measurable",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.09, "Gradient-boosted models, 5-fold cross-validation grouped by residue (repeated 5 times), 3,325 substitutions (Giacomelli) and 1,147 (Kato). "
         "Analysis fixed in writing before running. Function, not stability.",
         ha="center", fontsize=8.8, color=INK2)
fig.savefig(HERE / "alphamissense_figure.png", bbox_inches="tight")
print("saved")
