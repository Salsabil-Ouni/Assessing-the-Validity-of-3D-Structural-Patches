"""Check B: equivalence framing for the existing "no added value" contrasts. No new data, no new model fit:
re-expresses already-saved bootstrap intervals as an approximate minimum detectable effect (MDE) at 80% power,
using the normal approximation SE = (CI_high - CI_low) / (2 * 1.96), MDE_80 = 2.80 * SE (one-sided z-test,
alpha=0.05, power=0.80: z_0.975 + z_0.80 = 1.96 + 0.84 = 2.80)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
Z = 1.96 + 0.84  # two-sided 95% CI + 80% power, one-sided detection

rows = []

def add(source, dataset, contrast, delta, lo, hi):
    se = (hi - lo) / (2 * 1.96)
    mde = Z * se
    rows.append(dict(source=source, dataset=dataset, contrast=contrast, point_estimate=round(delta, 4),
                     ci_low=round(lo, 4), ci_high=round(hi, 4), approx_SE=round(se, 4),
                     MDE_80pct_power=round(mde, 4)))

vm = pd.read_csv(ROOT / "variant_ml" / "results_contrasts.csv")
for _, r in vm.iterrows():
    add("variant_ml", r.assay, r.contrast, r.delta_spearman, r.ci_low, r.ci_high)

am = pd.read_csv(ROOT / "alphamissense_test" / "results_contrasts.csv")
for _, r in am[am.contrast.str.startswith("PRIMARY")].iterrows():
    add("alphamissense_test (function)", r.assay, r.contrast, r.delta_spearman, r.ci_low, r.ci_high)

ams = pd.read_csv(ROOT / "alphamissense_test" / "stability_contrasts.csv")
for _, r in ams[ams.contrast.str.startswith("PRIMARY")].iterrows():
    add("alphamissense_test (stability)", r.descriptor_version, r.contrast, r.delta_LOO_spearman, r.ci_low, r.ci_high)

sf = pd.read_csv(ROOT / "structure_ai_test" / "results_function_contrasts.csv")
for _, r in sf[sf.contrast.str.startswith("PRIMARY")].iterrows():
    add("structure_ai_test (function)", r.dataset, r.contrast, r.delta_spearman, r.ci_low, r.ci_high)

ss = pd.read_csv(ROOT / "structure_ai_test" / "results_stability_contrasts.csv")
for _, r in ss[ss.contrast.str.startswith("PRIMARY")].iterrows():
    add("structure_ai_test (stability)", "stability (43 variants)", r.contrast, r.delta_LOO_spearman, r.ci_low, r.ci_high)

# main-text Q1 odds ratio, on the log scale (95% CI [1.5, 59])
lo, hi = np.log(1.5), np.log(59)
se_logOR = (hi - lo) / (2 * 1.96)
mde_logOR = Z * se_logOR
rows.append(dict(source="main thesis Q1", dataset="n=37", contrast="patch OR (log scale)", point_estimate=round(np.log(8.0), 4),
                 ci_low=round(lo, 4), ci_high=round(hi, 4), approx_SE=round(se_logOR, 4), MDE_80pct_power=round(mde_logOR, 4)))
print(f"Q1 odds ratio, log scale: point log(OR)={np.log(8.0):.3f}, approx SE={se_logOR:.3f}, "
      f"MDE at 80% power = {mde_logOR:.3f} log-OR units, i.e. an OR of only {np.exp(mde_logOR):.2f} would be "
      f"reliably detectable against a null of OR=1 with n=37 at this power.")

out = pd.DataFrame(rows)
out.to_csv(HERE / "resultsB_equivalence.csv", index=False)
print(f"\n{len(out)} primary contrasts re-expressed. Full table:")
print(out.to_string(index=False))
print("\nReading: MDE_80pct_power is the smallest TRUE gain in Spearman correlation that this sample size and "
      "bootstrap procedure would have detected 80% of the time. Point estimates below this value are "
      "uninformative about whether a smaller true effect exists; they do not demonstrate its absence.")
