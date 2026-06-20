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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 300,
})

categories = [
    "Sequence & Variant Analysis",
    "Drug-Target Discovery",
    "Literature & Clinical Mining",
    "Transcriptomics & Single-Cell",
    "Structural & Protein Biology",
    "Pathway & Systems Biology",
]

systems = ["BioClaw (PLEAS)", "Non-PLEAS Baseline", "GPT-5.5-thinking-ext."]
sys_short = ["BioClaw", "Baseline", "GPT-5.5"]

scores = np.array([
    [7, 6, 7],
    [6, 6, 4],
    [6, 8, 6],
    [7, 6, 2],
    [8, 7, 7],
    [8, 6, 5],
])

max_per_cat = 8
totals = scores.sum(axis=0)
max_total = 48
fractions = scores / max_per_cat

cmap = mcolors.LinearSegmentedColormap.from_list(
    "bench", ["#c62828", "#ef6c00", "#f9a825", "#9ccc65", "#43a047", "#1b5e20"]
)
norm = mcolors.Normalize(vmin=0, vmax=1)

fig = plt.figure(figsize=(8.5, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)
ax_heat = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])

for i in range(len(categories)):
    for j in range(len(systems)):
        val = fractions[i, j]
        color = cmap(norm(val))
        rect = plt.Rectangle(
            (j, i), 1, 1, facecolor=color,
            edgecolor="white", linewidth=2.5, clip_on=False,
        )
        ax_heat.add_patch(rect)
        pct = int(round(val * 100))
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if lum < 0.45 else "#1a1a1a"
        ax_heat.text(
            j + 0.5, i + 0.5,
            f"{scores[i, j]}/{max_per_cat}",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=text_color,
        )
        ax_heat.text(
            j + 0.5, i + 0.74,
            f"({pct}%)",
            ha="center", va="center", fontsize=7,
            color=text_color, alpha=0.85,
        )

ax_heat.set_xlim(0, len(systems))
ax_heat.set_ylim(0, len(categories))
ax_heat.set_xticks([x + 0.5 for x in range(len(systems))])
ax_heat.set_xticklabels(sys_short, fontsize=9, fontweight="bold")
ax_heat.xaxis.set_ticks_position("top")
ax_heat.xaxis.set_label_position("top")
ax_heat.set_yticks([y + 0.5 for y in range(len(categories))])
ax_heat.set_yticklabels(categories, fontsize=8.5)
ax_heat.invert_yaxis()
ax_heat.tick_params(axis="both", which="both", length=0, pad=6)
for spine in ax_heat.spines.values():
    spine.set_visible(False)
ax_heat.set_title("Per-Category Scores (out of 8)", fontsize=10, fontweight="bold", pad=14)

bar_colors = ["#1b5e20", "#43a047", "#ef6c00"]
y_pos = np.arange(len(systems))
bars = ax_bar.barh(
    y_pos, totals, color=bar_colors,
    edgecolor="none", height=0.55, zorder=3,
)

for idx, (bar, total) in enumerate(zip(bars, totals)):
    pct = round(total / max_total * 100, 1)
    ax_bar.text(
        total + 0.8, bar.get_y() + bar.get_height() / 2,
        f"{total}/{max_total} ({pct}%)",
        ha="left", va="center", fontsize=8.5, fontweight="bold",
    )

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(sys_short, fontsize=9, fontweight="bold")
ax_bar.set_xlim(0, 56)
ax_bar.set_xlabel("Total Score (out of 48)", fontsize=8)
ax_bar.set_title("Overall", fontsize=10, fontweight="bold", pad=14)
ax_bar.invert_yaxis()
ax_bar.tick_params(axis="y", which="both", length=0, pad=6)
ax_bar.tick_params(axis="x", which="both", labelsize=7.5)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_visible(False)
ax_bar.grid(axis="x", color="#e0e0e0", linewidth=0.5, zorder=0)

fig.text(
    0.5, -0.01,
    "Fig. 2.  24-task biomedical benchmark results. "
    "Scored by blinded multi-judge evaluation\n"
    "(3 independent LLM judges, majority-vote aggregation).",
    ha="center", va="top", fontsize=8, style="italic", color="#333333",
)

plt.savefig(
    "/home/user/PLEAS/case_studies/figures/fig2_benchmark.png",
    dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15,
)
plt.savefig(
    "/home/user/PLEAS/case_studies/figures/fig2_benchmark.pdf",
    bbox_inches="tight", facecolor="white", pad_inches=0.15,
)
print("Saved fig2_benchmark.png and fig2_benchmark.pdf")
