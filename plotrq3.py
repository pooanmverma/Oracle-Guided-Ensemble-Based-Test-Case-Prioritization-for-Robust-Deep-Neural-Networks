"""
RQ3 guidance curves: accuracy vs retraining budget (1-10%).
Ensemble metrics solid, confidence metrics dashed.
Run on the server where the CSVs live; saves rq3_guidance.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "MNIST":   f"{BASE}/retrained_models/guidance_results.csv",
    "Fashion": f"{BASE}/retrained_models_fashion/guidance_results_fashion.csv",
    "SVHN":    f"{BASE}/retrained_models_svhn/guidance_results_svhn.csv",
}

ENSEMBLE = ["ensgini", "disagreement", "variance", "mutualinfo"]
CONFIDENCE = ["deepgini", "softmax", "entropy", "pcs"]

ENS_COLORS = {"ensgini": "#1b5e20", "disagreement": "#2e7d32",
              "variance": "#388e3c", "mutualinfo": "#43a047"}
CONF_COLORS = {"deepgini": "#b71c1c", "softmax": "#c62828",
               "entropy": "#d32f2f", "pcs": "#e53935"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

for ax, (name, csv) in zip(axes, DATASETS.items()):
    df = pd.read_csv(csv)
    piv = df.groupby(["Metric", "TopPercent"])["Accuracy"].mean().unstack()
    x = piv.columns.tolist()

    for m in ENSEMBLE:
        if m in piv.index:
            ax.plot(x, piv.loc[m], "-", color=ENS_COLORS[m],
                    linewidth=2, marker="o", markersize=3, label=m)
    for m in CONFIDENCE:
        if m in piv.index:
            ax.plot(x, piv.loc[m], "--", color=CONF_COLORS[m],
                    linewidth=1.5, marker="s", markersize=3, label=m)

    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Prioritized tests added (%)")
    ax.set_ylabel("Accuracy on V")
    ax.grid(alpha=0.3)
    ax.set_xticks(x)

# single shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=8,
           bbox_to_anchor=(0.5, -0.08), fontsize=9)

plt.tight_layout()
plt.savefig(f"{BASE}/rq3_guidance.png", dpi=200, bbox_inches="tight")
print("saved ->", f"{BASE}/rq3_guidance.png")