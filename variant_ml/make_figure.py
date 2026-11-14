"""Figure for the variant-level test (dark ML style), read directly from the saved result tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SURF, INK, INK2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#34332f"
BLUE, ORANGE = "#3987e5", "#d95926"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
mod = pd.read_csv(HERE / "results_models.csv")
con = pd.read_csv(HERE / "results_contrasts.csv")
imp = pd.read_csv(HERE / "results_importance.csv")
ASSAYS = [("Giacomelli 2018", BLUE), ("Kato 2003", ORANGE)]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=170, gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.0], wspace=0.75))

# A: accuracy by feature set
ax = axes[0]
sets = list(mod.features.unique())[::-1]
y = np.arange(len(sets))
for j, (a, c) in enumerate(ASSAYS):
    v = [mod[(mod.assay == a) & (mod.features == s)].oof_spearman.iloc[0] for s in sets]
    ax.barh(y + (j - 0.5) * 0.36, v, height=0.32, color=c, edgecolor=SURF, linewidth=1.5)
    for yy, vv in zip(y + (j - 0.5) * 0.36, v):
        ax.text(vv + 0.006, yy, f"{vv:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(sets, fontsize=9.5)
ax.set_xlim(0.45, 0.70); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Held-out Spearman with measured damage", fontsize=9.5)
ax.set_title("A. Burial helps, the patch score barely moves it", loc="left", fontsize=11.5, fontweight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=c, label=a) for a, c in ASSAYS], loc="upper center", bbox_to_anchor=(0.5, -0.11),
          ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2)

# B: forest of the two primary contrasts
ax = axes[1]
rows = []
for a, c in ASSAYS:
    for _, r in con[con.assay == a].iterrows():
        short = "patch on top of\nsubstitution + burial" if "Contrast 2" in r.contrast else \
                "patch on top of\nsubstitution + burial + own"
        rows.append((f"{a.split()[0]}\n{short}", r.delta_spearman, r.ci_low, r.ci_high, c))
rows = rows[::-1]
for i, (lab, d, lo, hi, c) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round")
    ax.scatter([d], [i], s=90, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.text(d, i + 0.2, f"{d:+.3f}", ha="center", fontsize=9, color=INK2)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(-0.04, 0.04); ax.set_ylim(-0.6, len(rows) - 0.4); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Gain in held-out Spearman from adding the patch score\n(dot = estimate, bar = 95% interval over residues)", fontsize=9)
ax.set_title("B. Every interval includes zero", loc="left", fontsize=11.5, fontweight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

# C: importance
ax = axes[2]
blocks = ["ESM-2 zero-shot score", "burial (RSA)", "amino-acid chemistry", "own alanine-scan value", "patch score"]
yb = np.arange(len(blocks))[::-1]
for j, (a, c) in enumerate(ASSAYS):
    v = [imp[(imp.assay == a) & (imp.feature_block == b)].spearman_drop.iloc[0] for b in blocks]
    ax.barh(yb + (j - 0.5) * 0.36, v, height=0.32, color=c, edgecolor=SURF, linewidth=1.5)
    for yy, vv in zip(yb + (j - 0.5) * 0.36, v):
        ax.text(vv + 0.005, yy, f"{vv:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(yb); ax.set_yticklabels(blocks, fontsize=9.5)
ax.set_xlim(0, 0.36); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Drop in accuracy when the input is scrambled", fontsize=9.5)
ax.set_title("C. What the model relies on", loc="left", fontsize=11.5, fontweight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.suptitle("Variant-level machine learning on the functional datasets: the patch score adds nothing measurable beyond burial",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.08,"Gradient-boosted models, 5-fold cross-validation grouped by residue (repeated 5 times), 3,325 substitutions (Giacomelli) and 1,147 (Kato). "
         "Analysis fixed in writing before running. Function, not stability. Substitution inputs include an ESM-2 zero-shot score.",
         ha="center", fontsize=8.8, color=INK2)
fig.savefig(HERE / "variant_ml_figure.png", bbox_inches="tight")
print("saved")
