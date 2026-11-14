"""
Figure for the hotspot-threshold sensitivity sweep (hotspot_threshold_sweep.py).
Companion to sensitivity_panel.png (which sweeps the destabilisation threshold
and the neighbour-distance cut-off, but never the hotspot threshold itself).

Left panel: Q1 odds ratio (95% CI) vs hotspot cut-off g_i, marginal vs buried-only.
Right panel: number of hotspot residues (bars) vs the same x-axis, showing how
fast the "hotspot" label loses meaning as the cut-off is relaxed.
"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "results" / "hotspot_threshold_sweep.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
lo = df.Q1_OR - df.Q1_CI_low
hi = df.Q1_CI_high - df.Q1_OR
ax.errorbar(df.hotspot_cutoff, df.Q1_OR, yerr=[lo, hi], fmt="o-", color="#1f4e8c",
            capsize=4, ms=6, label="Q1 OR (marginal)")
buried = df.dropna(subset=["buried_only_OR"])
ax.plot(buried.hotspot_cutoff, buried.buried_only_OR, "s--", color="#d62728", ms=6,
        label="Q1 OR (buried-only)")
ax.axhline(1.0, color="grey", ls=":", lw=1)
ax.axvline(2.0, color="grey", ls=":", lw=1, label="pipeline default (2.0)")
ax.set_yscale("log")
ax.set_xlabel(r"hotspot cut-off $g_i$ (kcal/mol)")
ax.set_ylabel("Q1 odds ratio (95% CI)")
ax.set_title("Enrichment vs hotspot threshold")
ax.legend(fontsize=8)

ax = axes[1]
ax.bar(df.hotspot_cutoff, df.n_hotspots, width=0.35, color="#1f4e8c", alpha=0.8)
ax.axvline(2.0, color="grey", ls=":", lw=1)
ax.set_xlabel(r"hotspot cut-off $g_i$ (kcal/mol)")
ax.set_ylabel("number of hotspot residues (of 194)")
ax.set_title("Hotspot count vs threshold")
for x, y in zip(df.hotspot_cutoff, df.n_hotspots):
    ax.text(x, y + 2, str(int(y)), ha="center", fontsize=8)

fig.suptitle("Robustness of Q1 to the hotspot-definition threshold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "hotspot_threshold_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: results/figures/hotspot_threshold_panel.png")
