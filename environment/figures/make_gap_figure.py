#!/usr/bin/env python3
"""Figure: campaign_summary.json's per-judge, per-seed rank-1/rank-2 gap vs the null-model
noise floor.

Data-driven, read directly from campaign_summary.json (3 seeds x 3 instrument-valid judges =
9 points), not from memory or from the prose in GOAI_SEMIFINAL_PROBLEM_DOC.md section 4.

WHAT THE NUMBERS ACTUALLY SHOW, verified here rather than assumed: for every one of the 9
(judge, seed) pairs, the observed final_gap_by_judge value is BELOW that judge's own
null_model_by_judge gap_p95 for that seed -- none of the 9 technically "beats chance" by the
environment's own null_model_by_judge() rule ("an observed gap must exceed gap_p95"). What
does distinguish them is how CLOSE the observed gap gets to its own threshold:

  gemma4-12b:            29% / 64% / 71% of its seed's p95   (seeds 0, 1, 42)
  gemma4-e4b:              2% /  3% / 18% of its seed's p95
  olmo3-7b:                2% / 15% / 13% of its seed's p95

gemma4-12b consistently reaches 3-4x closer to its noise floor than the other two judges,
which never clear ~18%. That gradient -- not a binary pass/fail against p95 -- is the honest
version of "gemma4-12b carries the real signal; gemma4-e4b/olmo3-7b sit near the noise floor"
from Finding 2, and it is drawn explicitly so a reader does not have to take the pass/fail
framing in the prose on faith. Bars are the observed gap; diamonds are that seed's own
null-model p95 threshold, plotted at the same x position so the shortfall is visible directly
rather than asserted.

Matches the house palette used in JUDGe-2026/figures/make_env_figure.py and
make_loop_figure.py in this directory.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_campaign_gaps.png")
DATA = os.path.join(os.path.dirname(HERE), "campaign_summary.json")

INK, ACCENT, MUTED, EXCL = "#1a1a1a", "#0B6E4F", "#8a8a8a", "#B23A48"

campaign = json.load(open(DATA))

JUDGES = ["gemma4-12b", "gemma4-e4b", "olmo3-7b"]
JUDGE_COLOR = {"gemma4-12b": ACCENT, "gemma4-e4b": MUTED, "olmo3-7b": "#5B7DB1"}
SEEDS = [row["seed"] for row in campaign]

# pull gap and p95 threshold per (judge, seed) straight from the file
gap = {j: [] for j in JUDGES}
p95 = {j: [] for j in JUDGES}
rank1 = {j: [] for j in JUDGES}
for row in campaign:
    fg = row["final_gap_by_judge"]
    nm = row["null_model_by_judge"]
    for j in JUDGES:
        gap[j].append(fg[j]["gap"])
        rank1[j].append(fg[j]["rank1"])
        p95[j].append(nm[j]["gap_p95"])

plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(9.8, 6.2))

n_seed = len(SEEDS)
group_w = 0.72
bar_w = group_w / n_seed
x_base = np.arange(len(JUDGES))

for si, seed in enumerate(SEEDS):
    xs = x_base - group_w / 2 + bar_w * (si + 0.5)
    heights = [gap[j][si] for j in JUDGES]
    colors = [JUDGE_COLOR[j] for j in JUDGES]
    bars = ax.bar(xs, heights, width=bar_w * 0.92, color=colors,
                   edgecolor=INK, linewidth=0.9, zorder=3,
                   alpha=0.55 + 0.15 * si)  # slightly darker per later seed, same hue family
    # null-model p95 threshold for this exact (judge, seed), same x position
    thresholds = [p95[j][si] for j in JUDGES]
    ax.scatter(xs, thresholds, marker="D", s=46, facecolor="white",
               edgecolor=INK, linewidth=1.3, zorder=5)
    for x, h, t in zip(xs, heights, thresholds):
        ax.plot([x, x], [h, t], color=INK, linewidth=0.9, linestyle=":", zorder=4)
    # seed + rank-1 operator labels under each bar
    for x, j in zip(xs, JUDGES):
        ax.text(x, -0.16, f"seed {seed}", ha="center", va="top", fontsize=6.6, color=MUTED,
                rotation=0)
        ax.text(x, heights[JUDGES.index(j)] + 0.06, rank1[j][si].replace("_", "-"),
                ha="center", va="bottom", fontsize=6.6, color=INK, rotation=90, zorder=6)

ax.set_xticks(x_base)
ax.set_xticklabels(JUDGES, fontsize=10.5, weight="bold")
ax.set_ylabel("rank-1 to rank-2 operator gap (mean delta, judge's own scale)", fontsize=8.8)
ax.set_ylim(-0.55, 3.4)
ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
ax.spines[["top", "right"]].set_visible(False)

# legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_handles = [
    Patch(facecolor=ACCENT, edgecolor=INK, alpha=0.7, label="observed gap (bar height)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
           markeredgecolor=INK, markersize=7,
           label="that seed's null-model gap_p95 (must be exceeded to beat chance)"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=7.6, frameon=False)

fig.suptitle("Campaign gaps vs. the null-model noise floor, 3 seeds x 3 judges",
             fontsize=12.5, weight="bold", y=0.985)
ax.set_title("no (judge, seed) pair clears its own p95 line -- but gemma4-12b's gap reaches\n"
             "29-71% of it, vs. 2-18% for gemma4-e4b/olmo3-7b",
             fontsize=8.4, style="italic", color=MUTED, pad=12)

fig.tight_layout(rect=[0, 0.03, 1, 0.90])
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)

# print the ratios used in the caption, so they are checkable against this run's output
print("\nobserved gap as % of that seed's null-model p95:")
for j in JUDGES:
    pct = [round(100 * g / t, 1) for g, t in zip(gap[j], p95[j])]
    print(f"  {j:14} {pct}")
