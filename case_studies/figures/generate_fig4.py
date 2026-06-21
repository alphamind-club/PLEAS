"""
Generate Fig. 4 for IEEE IRI 2026 paper.
Call-by-call input token counts: Baseline vs BioPLEAS.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 300,
})

baseline = [23591, 46355, 51614, 57984, 67563, 75242, 83565, 91642,
            98577, 106071, 113217, 123906, 130879, 138003, 144429, 151694, 169117, 195498]

biopleas = [19896, 22101, 24572, 26506, 28722, 30262, 32009, 33782,
            35821, 37029, 38754, 40504, 41687, 43176, 44624, 46555, 48980, 51845]

calls_bl = np.arange(1, len(baseline) + 1)
calls_bp = np.arange(1, len(biopleas) + 1)

fig, ax = plt.subplots(figsize=(6, 3.2))

ax.plot(calls_bl, [t / 1000 for t in baseline], "o-",
        color="#d32f2f", markersize=4, linewidth=1.8, label="Baseline (no PLEAS)")
ax.plot(calls_bp, [t / 1000 for t in biopleas], "s-",
        color="#1b5e20", markersize=4, linewidth=1.8, label="BioPLEAS (full PLEAS)")

ax.set_xlabel("API Call Index")
ax.set_ylabel("Input Tokens (thousands)")
ax.set_xlim(0.5, 18.5)
ax.set_xticks(range(1, 19))
ax.set_ylim(0, 210)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bl_avg = sum(baseline) / len(baseline) / 1000
bp_avg = sum(biopleas) / len(biopleas) / 1000
ax.axhline(bl_avg, color="#d32f2f", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(bp_avg, color="#1b5e20", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(18.3, bl_avg, f"avg {bl_avg:.0f}K", fontsize=7, color="#d32f2f", va="center")
ax.text(18.3, bp_avg, f"avg {bp_avg:.0f}K", fontsize=7, color="#1b5e20", va="center")

fig.text(
    0.5, -0.02,
    "Fig. 4.  Call-by-call input token counts.\n"
    "Both conditions increase over time; baseline context grows rapidly while BioPLEAS grows slowly under phase isolation and trace compression.",
    ha="center", va="top", fontsize=8, style="italic", color="#333333",
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("/home/user/PLEAS/case_studies/figures/fig4_tokens.png",
            dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.savefig("/home/user/PLEAS/case_studies/figures/fig4_tokens.pdf",
            bbox_inches="tight", facecolor="white", pad_inches=0.15)
print("Saved fig4_tokens.png and fig4_tokens.pdf")
