#!/usr/bin/env python3
"""Sweep of UCB1's exploration constant c, at the official budget and seeds.

WHY THIS RUN EXISTS

The write-up's strongest claim about this environment is that setting UCB1's
exploration constant to zero reproduces greedy_agent's collapse -- 0 of 9
(judge, seed) cells clearing their own null-model p95 -- from a completely
independent code path, while every c > 0 clears several. That claim had no
committed artifact behind it. ucb_summary.json holds three runs, all at
c=1.0, and run_ucb_baseline.py hardcoded c with no way to override it, so a
reviewer reading the c-sweep table could not re-derive a single row of it from
this repository. This script is that artifact.

SAME TERMS AS THE COMMITTED RUNS

Budget, rounds, seeds and the judge panel are imported from run_campaign.py,
not restated, so every row here is on the same terms as campaign_summary.json,
random_baseline_summary.json and ucb_summary.json. The only thing that varies
across rows is c. clears() and alloc_from_cells() are imported from
run_budget_sweep.py rather than reimplemented, so the rows are scored by
exactly the code that scored the committed budget sweep.

The c=1.0 row is re-run here rather than cited, because re-running it is the
check that matters: if this script's c=1.0 row does not match the committed
ucb_summary.json cell for cell, then something about the sweep is not on the
terms it claims to be on. That comparison is performed and recorded under
"reference_check", and a mismatch is reported rather than smoothed over.

WHAT IT DOES NOT TOUCH

Nothing here writes to campaign_summary.json, random_baseline_summary.json,
ucb_summary.json, budget_sweep_summary.json, exploration_log.jsonl or any
exploration_log_{campaign,random,ucb}_seed*.jsonl. Every per-step log this
script writes goes under environment/sweep_logs/ with the c in its filename,
including the c=1.0 row, and the only summary it writes is
ucb_c_sweep_summary.json.

METRIC DEFINITIONS, STATED SO THE TABLE CAN BE CHECKED

  clears        a (judge, seed) cell counts as clearing when that judge's
                observed rank1-vs-rank2 gap exceeds its own baseline_null_model
                gap_p95 in that run. 3 judges x 3 seeds = 9 cells per c.
  mean gap      mean of the observed gaps over those 9 cells.
  skew          max_n / min_n across the six operators for a judge, then
                averaged over the 9 cells. Reported as a mean because that is
                what the write-up's table reports; the median is recorded
                alongside it, since the two differ noticeably for lopsided
                allocations and a reader should not have to guess which is
                which.
  rank-1 =      count of the 9 cells whose top-ranked operator is euphemism.
  euphemism

Usage: python3 run_ucb_c_sweep.py
Output: ucb_c_sweep_summary.json
        sweep_logs/exploration_log_ucb_c<tag>_seed<seed>.jsonl
"""
import json
import os
import statistics as st

from env import MOVES
from run_budget_sweep import alloc_from_cells, clears
from run_campaign import BUDGET, SEEDS
from run_ucb_baseline import SWEEP_LOG_DIR, c_tag, run_seed

HERE = os.path.dirname(os.path.abspath(__file__))

# The six values the write-up's table reports. 0.0 is the load-bearing one: it
# switches the exploration bonus off entirely, leaving pure exploitation.
C_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

# The committed c=1.0 artifact, read back only to check this sweep reproduces
# it. It is an input here and never an output.
REFERENCE_C = 1.0
REFERENCE_FILE = "ucb_summary.json"

OUT_PATH = os.path.join(HERE, "ucb_c_sweep_summary.json")


def rank1_euphemism(summaries):
    """How many of the 9 (judge, seed) cells put euphemism first. This is the
    substantive finding the whole environment exists to measure, so it is
    tracked per c: a policy that clears its null but ranks a different operator
    first has not recovered the effect, it has found a different one."""
    hits = total = 0
    for s in summaries:
        for g in (s.get("final_gap_by_judge") or {}).values():
            if not g:
                continue
            total += 1
            if g["rank1"] == "euphemism":
                hits += 1
    return hits, total


def aggregate(c, summaries):
    gaps, p95s, skews, max_ns, min_ns = [], [], [], [], []
    cleared = undetermined = total = 0
    missing_operator = 0

    for s in summaries:
        for ok in clears(s).values():
            total += 1
            if ok is None:
                undetermined += 1
            elif ok:
                cleared += 1
        for g in (s.get("final_gap_by_judge") or {}).values():
            if g:
                gaps.append(g["gap"])
        for nb in (s.get("null_model_by_judge") or {}).values():
            p95s.append(nb["gap_p95"])
        for a in alloc_from_cells(s["cells"], sorted(s["final_gap_by_judge"])).values():
            max_ns.append(a["max_n"])
            min_ns.append(a["min_n"])
            if a["operators_covered"] < len(MOVES):
                missing_operator += 1
            if a["skew_max_over_min"] is not None:
                skews.append(a["skew_max_over_min"])

    hits, euph_total = rank1_euphemism(summaries)
    return {
        "ucb_c": c,
        "cells_clearing_null_p95": f"{cleared}/{total}",
        "cells_cleared": cleared,
        "cells_total": total,
        "cells_undetermined": undetermined,
        "mean_gap": round(st.mean(gaps), 3) if gaps else None,
        "mean_null_p95": round(st.mean(p95s), 3) if p95s else None,
        "mean_skew_max_over_min": round(st.mean(skews), 1) if len(skews) == total else None,
        "median_skew_max_over_min": (round(st.median(skews), 2)
                                     if len(skews) == total else None),
        "skew_undefined_cells": total - len(skews),
        "mean_max_n": round(st.mean(max_ns), 1) if max_ns else None,
        "mean_min_n": round(st.mean(min_ns), 1) if min_ns else None,
        "rank1_euphemism": f"{hits}/{euph_total}",
        "cells_missing_an_operator": missing_operator,
    }


def reference_check(swept):
    """Does this script's c=1.0 row reproduce the committed ucb_summary.json?

    Compared on the substance -- per-cell n and mean_delta, the final gap per
    judge, and the null-model p95 per judge -- and deliberately not on
    log_path, which differs by design because this script keeps its logs out of
    the committed ones' way."""
    path = os.path.join(HERE, REFERENCE_FILE)
    if not os.path.exists(path):
        return {"checked": False, "reason": f"{REFERENCE_FILE} not present"}
    committed = json.load(open(path))
    mine = {s["seed"]: s for s in swept}
    diffs = []
    for ref in committed:
        got = mine.get(ref["seed"])
        if got is None:
            diffs.append(f"seed {ref['seed']} missing from the sweep")
            continue
        for field in ("cells", "final_gap_by_judge", "null_model_by_judge",
                      "steps_taken", "rejections"):
            if got[field] != ref[field]:
                diffs.append(f"seed {ref['seed']}: {field} differs")
    return {"checked": True,
            "reference_file": REFERENCE_FILE,
            "reference_c": REFERENCE_C,
            "matches": not diffs,
            "differences": diffs,
            "note": ("the c=1.0 row of this sweep is a fresh run, so a match "
                     "means the sweep is on the same terms as the committed "
                     "UCB baseline and the environment is still deterministic "
                     "under a fixed seed")}


def main():
    os.makedirs(SWEEP_LOG_DIR, exist_ok=True)
    rows, runs, ref_runs = [], [], None

    for c in C_VALUES:
        summaries = []
        for seed in SEEDS:
            # Explicit log path for every c, including the default 1.0, so the
            # committed exploration_log_ucb_seed*.jsonl files are never opened.
            log_path = os.path.join(
                SWEEP_LOG_DIR, f"exploration_log_ucb_{c_tag(c)}_seed{seed}.jsonl")
            s = run_seed(seed, c=c, log_path=log_path)
            s["clears_null_p95_by_judge"] = clears(s)
            summaries.append(s)
            print(f"c={c:<5g} seed={seed:<3} steps={s['steps_taken']:<5} "
                  f"stopped={s['stopped_reason']:<17} "
                  f"clears={s['clears_null_p95_by_judge']}")
        if c == REFERENCE_C:
            ref_runs = summaries
        rows.append(aggregate(c, summaries))
        runs.extend(summaries)

    out = {
        "what": "UCB1 exploration constant sweep at the official budget and seeds",
        "why": ("the write-up's c-sweep table had no committed artifact: "
                "ucb_summary.json holds three runs, all at c=1.0. This file is "
                "that table's source."),
        "budget": BUDGET,
        "seeds": SEEDS,
        "c_values": C_VALUES,
        "policy": "ucb_agent",
        "metrics": {
            "cells_clearing_null_p95": "observed gap > that judge's own baseline_null_model gap_p95, over 3 judges x 3 seeds",
            "mean_gap": "mean observed rank1-vs-rank2 gap over the same 9 cells",
            "mean_skew_max_over_min": "mean over the 9 cells of max_n / min_n across the six operators",
            "rank1_euphemism": "cells whose top-ranked operator is euphemism",
        },
        "comparison_table": rows,
        "reference_check": reference_check(ref_runs or []),
        "runs": runs,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")

    print(f"\n{'c':>5} {'clears':>8} {'mean_gap':>9} {'mean_p95':>9} "
          f"{'skew':>8} {'med_skew':>9} {'rank1=euph':>11}")
    for r in rows:
        print(f"{r['ucb_c']:>5g} {r['cells_clearing_null_p95']:>8} "
              f"{str(r['mean_gap']):>9} {str(r['mean_null_p95']):>9} "
              f"{str(r['mean_skew_max_over_min']):>8} "
              f"{str(r['median_skew_max_over_min']):>9} "
              f"{r['rank1_euphemism']:>11}")

    rc = out["reference_check"]
    if rc.get("checked"):
        print(f"\nc={REFERENCE_C:g} vs committed {REFERENCE_FILE}: "
              f"{'MATCHES' if rc['matches'] else 'DIFFERS'}")
        for d in rc["differences"]:
            print(f"  {d}")


if __name__ == "__main__":
    main()
