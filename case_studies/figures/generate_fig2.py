"""
Generate Fig. 2 for IEEE IRI 2026 paper.
Heatmap (left) + horizontal bar chart (right) showing
24-task biomedical benchmark results with blinded multi-judge scoring.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

categories = [
    "Sequence &\nVariant Analysis",
    "Drug-Target\nDiscovery",
    "Literature &\nClinical Mining",
    "Transcriptomics &\nSingle-Cell",
    "Structural &\nProtein Biology",
    "Pathway &\nSystems Biology",
]

systems = ["BioClaw\n(PLEAS)", "Non-PLEAS\nBaseline", "GPT-5.5-\nthinking-ext."]

scores = np.array([
    [7, 6, 7],  # Seq & Variant
    [6, 6, 4],  # Drug-Target
    [6, 8, 6],  # Lit & Clinical
    [7, 6, 2],  # Transcriptomics
    [8, 7, 7],  # Structural
    [8, 6, 5],  # Pathway
])

max_per_cat = 8
totals = scores.sum(axis=0)  # [42, 39, 31]
max_total = 48

fractions = scores / max_per_cat

cmap = mcolors.LinearSegmentedColormap.from_list(
    "bench", ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
)
norm = mcolors.Normalize(vmin=0, vmax=1)

fig, (ax_heat, ax_bar) = plt.subplots(
    1, 2, figsize=(10.5, 4.2),
    gridspec_kw={"width_ratios": [3.2, 1.2], "wspace": 0.15},
)

for i in range(len(categories)):
    for j in range(len(systems)):
        val = fractions[i, j]
        color = cmap(norm(val))
        ax_heat.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
        pct = int(round(scores[i, j] / max_per_cat * 100))
        text_color = "white" if val < 0.4 else "black"
        ax_heat.text(
            j + 0.5, i + 0.5,
            f"{scores[i, j]}/{max_per_cat} ({pct}%)",
            ha="center", va="center", fontsize=9, fontweight="bold", color=text_color,
        )

ax_heat.set_xlim(0, len(systems))
ax_heat.set_ylim(0, len(categories))
ax_heat.set_xticks([x + 0.5 for x in range(len(systems))])
ax_heat.set_xticklabels(systems, fontsize=9, fontweight="bold")
ax_heat.set_yticks([y + 0.5 for y in range(len(categories))])
ax_heat.set_yticklabels(categories, fontsize=8.5)
ax_heat.invert_yaxis()
ax_heat.tick_params(axis="both", which="both", length=0)
ax_heat.set_title("Per-Category Scores", fontsize=11, fontweight="bold", pad=10)

bar_colors = ["#1a9850", "#91cf60", "#fc8d59"]
bars = ax_bar.barh(
    range(len(systems)), totals, color=bar_colors,
    edgecolor="white", linewidth=1.5, height=0.6,
)

for idx, (bar, total) in enumerate(zip(bars, totals)):
    pct = round(total / max_total * 100, 1)
    ax_bar.text(
        total + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{total}/{max_total} ({pct}%)",
        ha="left", va="center", fontsize=9, fontweight="bold",
    )

ax_bar.set_yticks(range(len(systems)))
ax_bar.set_yticklabels(systems, fontsize=9, fontweight="bold")
ax_bar.set_xlim(0, 55)
ax_bar.set_xlabel("Total Score", fontsize=9)
ax_bar.set_title("Overall", fontsize=11, fontweight="bold", pad=10)
ax_bar.invert_yaxis()
ax_bar.tick_params(axis="y", which="both", length=0)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

fig.text(
    0.5, -0.02,
    "Fig. 2.  24-task biomedical scientific workflow benchmark results.\n"
    "Scored by blinded multi-judge evaluation (3 independent LLM judges, majority-vote aggregation).",
    ha="center", va="top", fontsize=9, style="italic",
)

fig.subplots_adjust(bottom=0.14, left=0.12, right=0.95)
plt.savefig("/home/user/PLEAS/case_studies/figures/fig2_benchmark.png", dpi=300, bbox_inches="tight")
plt.savefig("/home/user/PLEAS/case_studies/figures/fig2_benchmark.pdf", bbox_inches="tight")
print("Saved fig2_benchmark.png and fig2_benchmark.pdf")
