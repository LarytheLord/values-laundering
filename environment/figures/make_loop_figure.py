#!/usr/bin/env python3
"""Figure: the observe() -> step() -> feedback loop in env.py, drawn as boxes and arrows.

This is a code-level companion to JUDGe-2026/figures/fig4_environment.png (which draws the
FIXED/EXPLORABLE/FEEDBACK split as three static bands for the spec document). That figure
does not show the actual control flow inside Environment.observe()/step(); this one does,
so a reader who has not read env.py can see the loop close at a glance.

Traced directly from env.py:
  - Environment.__init__ applies the instrument-validity gate ONCE, at construction
    (judge kept in self.valid only if its mean literal score <= GATE=4.0); it is not
    re-applied every step, so it is drawn as a FIXED precondition, not a loop stage.
  - observe() reports: judges_valid / judges_excluded, per-cell coverage, unprobed cells,
    best_move_so_far, budget_left, rejections count.
  - the agent picks (judge, move, n) -- that choice is the EXPLORABLE surface: which of the
    6 MOVES, against which of the instrument-valid judges, at what sample size n.
  - step() pulls the pool of acts that judge covers for that move, shuffles, and for each of
    the first n: checks the OVERLAP_FLOOR (0.15) semantic-preservation gate on that specific
    rewrite. Below floor -> validator_verdict "reject", logged with its reason, no score.
    At/above floor -> ReplayJudge.score() returns literal/rewritten/delta/binary_pair, which
    updates self.results and is logged with validator_verdict "pass".
  - step() returns feedback: scored/rejected counts, per-act deltas, mean_delta,
    binary_flips, gap_to_second (via _gap(): rank-1 vs rank-2 operator mean, by judge),
    budget_left. Every record -- pass or reject -- is appended to the immutable JSONL log.
  - the next observe() call reads the updated self.results, so the agent's own past actions
    are what change what it sees next: that is the loop this figure exists to make visible.

Matches the house palette and box/arrow style used in JUDGe-2026/figures/make_env_figure.py
so the two figures read as one family rather than two different tools.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_env_loop.png")

INK, ACCENT, MUTED, EXCL = "#1a1a1a", "#0B6E4F", "#8a8a8a", "#B23A48"
FIXED_BG, EXPL_BG, FEED_BG, GATE_BG = "#EDEDEA", "#E4F1EA", "#FBECEE", "#F4F4F1"

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(10.4, 8.0))
ax.set_xlim(0, 112); ax.set_ylim(0, 100); ax.axis("off")

ax.text(56, 98.6, "env.py: the observe() -> step() -> feedback loop",
        ha="center", va="top", fontsize=12, weight="bold", color=INK)
ax.text(56, 95.6,
        "each loop turn is one call to greedy_agent()'s policy; the agent's own past steps change what the NEXT observe() sees",
        ha="center", va="top", fontsize=8.2, style="italic", color=MUTED)


def box(x, y, w, h, title, lines, bg, edge, title_color=None, title_size=9.2, title_gap=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.3",
                                linewidth=1.4, edgecolor=edge, facecolor=bg, zorder=2))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=title_size,
            weight="bold", color=title_color or edge, zorder=3)
    ax.text(x + w / 2, y + h - title_gap, "\n".join(lines), ha="center", va="top",
            fontsize=7.5, color=INK, linespacing=1.55, zorder=3)


def arrow(p1, p2, color, label=None, lw=1.8, rad=0.0, label_dx=0, label_dy=0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=lw, color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + label_dx, (p1[1] + p2[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.0, color=color,
                weight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


# ---- FIXED precondition, applied once at Environment.__init__, off to the left ----
box(1.5, 58.0, 25.0, 34.0, "FIXED (set at __init__)",
    ["act bank: frozen, single-",
     "domain, from kernel_payload.json",
     "",
     "judge panel: pinned versions,",
     "instrument-validity gate --",
     "kept in judges_valid only if",
     "mean literal score <= 4.0",
     "",
     "OVERLAP_FLOOR = 0.15,",
     "checked per rewrite in step()",
     "",
     "lineage separation: generator",
     "of a move is never its judge"],
    FIXED_BG, INK, title_size=8.6, title_gap=5.2)

# ---- main loop: 4 stacked boxes on the right, top to bottom, with a loop-back ----
LX, LW = 34.0, 62.0

box(LX, 82.0, LW, 13.0, "1. observe()",
    ["judges_valid / judges_excluded (gate already applied)",
     "coverage per (judge, move) cell, and which cells are unprobed",
     "best_move_so_far per judge  |  budget_left  |  rejections so far"],
    GATE_BG, INK)

arrow((LX + LW / 2, 82.0), (LX + LW / 2, 71.5), ACCENT,
      "agent reads state, picks a cell", label_dy=1.0)

box(LX, 58.5, LW, 13.0, "2. EXPLORABLE: agent's choice",
    ["which operator (1 of the 6 MOVES) to probe",
     "which judge, from judges_valid",
     "sample size n for this probe"],
    EXPL_BG, ACCENT)

arrow((LX + LW / 2, 58.5), (LX + LW / 2, 48.0), ACCENT,
      "step(judge, move, n)", label_dy=1.0)

box(LX, 34.0, LW, 14.0, "3. step(): apply the fixed gates",
    ["pool = acts this judge covers for this move, shuffled",
     "for each of the first n acts: overlap(act, rewrite) vs OVERLAP_FLOOR",
     "below floor -> reject          at/above floor -> ReplayJudge.score()"],
    GATE_BG, INK)

# branch: reject vs scored
arrow((LX + LW * 0.28, 34.0), (LX + LW * 0.12, 25.0), EXCL,
      "reject: logged with\nreason, no score", rad=-0.15, label_dx=-6.5, label_dy=1.5)
arrow((LX + LW * 0.72, 34.0), (LX + LW * 0.88, 25.0), ACCENT,
      "pass: literal, rewritten,\ndelta, binary pair", rad=0.15, label_dx=6.8, label_dy=1.5)

box(LX, 10.5, LW, 13.0, "4. FEEDBACK returned to the agent",
    ["per-act delta and mean_delta   |   binary_flips (did the verdict flip 1->0?)",
     "gap_to_second: rank-1 vs rank-2 operator mean, for THIS judge",
     "budget_left   |   every record above -- pass or reject -- appended to the immutable log"],
    FEED_BG, EXCL)

# loop back up to observe()
arrow((LX + LW, 17.0), (108.0, 17.0), INK, lw=1.6)
arrow((108.0, 17.0), (108.0, 88.5), INK, lw=1.6)
arrow((108.0, 88.5), (LX + LW, 88.5), INK, lw=1.6)
ax.text(110.6, 52.0, "next observe() sees the\nupdated results -- the loop closes",
        ha="center", va="center", fontsize=7.6, color=INK, weight="bold",
        rotation=90, zorder=5)

# FIXED feeds into both observe() (gate already baked into judges_valid) and step() (overlap floor)
arrow((26.0, 86.0), (34.0, 86.0), MUTED, "gates judges_valid", lw=1.5, label_dy=1.4)
arrow((26.0, 62.0), (34.0, 62.0), MUTED, "", lw=1.5)
arrow((26.0, 40.0), (34.0, 40.0), MUTED, "overlap floor\napplied here", lw=1.5, label_dy=1.6)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
