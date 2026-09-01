#!/usr/bin/env python3
"""Budget sweep: greedy_agent vs random_agent at budgets where the budget binds.

WHY THIS RUN EXISTS

At the official campaign budget of 2000 the exploration space is not scarce.
The space is 81 acts x 6 operators x 3 instrument-valid judges = 1458 probe
cells, so a budget of 2000 is larger than the space itself. run_campaign.py
(greedy) and run_random_baseline.py (random) are both run at 2000, and at that
budget the directed policy loses: greedy clears its own null-model p95 in 0 of
9 (judge, seed) cells while random clears in 6 of 9, because greedy's depth
phase concentrates several hundred probes on one operator and leaves the other
five at n=3-26, which raises its own significance threshold.

The open question that answers is: is that a fact about greedy, or a fact about
running greedy at a budget where nothing is scarce? A directed policy has no
job to do when probes are abundant. This script tests the same two policies at
budgets that actually bind -- 150, 300, 600 -- against the same 3 seeds
(0, 1, 42) and the same judge panel, so the comparison is like-for-like with
the committed 2000 runs.

WHAT IT DOES NOT TOUCH

Nothing here writes to campaign_summary.json, random_baseline_summary.json, or
any exploration_log*.jsonl in this directory. Those are the official results.
Every per-step log this script writes goes under environment/sweep_logs/ with a
"sweep" filename, and the only summary it writes is budget_sweep_summary.json.

The budget=2000 row of the comparison table is NOT re-run here: it is read back
from the two committed summary files, so the table cites the official numbers
rather than a second copy of them.

WHAT IT RECORDS

Per (policy, budget, seed), the same fields the two existing runners record --
per-cell n and mean_delta, final_gap_by_judge, and null_model_by_judge via
Environment.baseline_null_model -- plus, per (judge, seed), the allocation
spread (max n and min n across the six operators) because the allocation is the
mechanism that moves the null threshold.

Usage: python3 run_budget_sweep.py
Output: budget_sweep_summary.json
        sweep_logs/exploration_log_sweep_<policy>_b<budget>_seed<seed>.jsonl
"""
import json
import os
import statistics as st

from env import Bank, Environment, greedy_agent, random_agent, MOVES
from run_campaign import load_judges, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "sweep_logs")

# 1458 probe cells in the space; the official campaign budget is 2000, i.e. not
# binding. These three are.
BUDGETS = [150, 300, 600]

POLICIES = {"greedy": greedy_agent, "random": random_agent}

# The already-committed budget=2000 results, cited rather than re-run.
REFERENCE_BUDGET = 2000
REFERENCE_FILES = {"greedy": "campaign_summary.json",
                   "random": "random_baseline_summary.json"}


def rounds_for(budget):
    """A round count that guarantees the BUDGET, not the round cap, is what
    stops the agent. Both agents call env.step(..., n=6), and a step always
    consumes at least 1 probe when any budget is left, so `budget` rounds can
    never be reached before the budget is gone. Both agents also break out of
    their loop as soon as budget_left hits 0, so an over-large cap costs
    nothing."""
    return budget


def cell_table(env):
    """Per-(judge, operator) n / mean / stdev, in the same shape and iteration
    order run_campaign.py and run_random_baseline.py use."""
    cells = {}
    for jk in env.valid:
        for m in MOVES:
            v = env.results.get((jk, m), [])
            if v:
                cells[f"{jk}|{m}"] = {"n": len(v), "mean_delta": round(st.mean(v), 3),
                                      "stdev": round(st.pstdev(v), 3) if len(v) > 1 else 0.0}
    return cells


def null_table(env):
    """baseline_null_model per valid judge, same call and same order as the two
    official runners. This is the point of the comparison: the shuffle conditions
    on the REALIZED per-operator sample sizes, so a policy that piles its budget
    onto one operator raises its own significance bar."""
    out = {}
    for jk in env.valid:
        nb = env.baseline_null_model(jk)
        if "gap_p95" in nb:
            out[jk] = nb
    return out


def allocation_table(env):
    """Per judge, how lopsided the realized allocation across the six operators
    is. skew = max_n / min_n, or null when some operator was never probed at all
    (an unprobed operator makes the ratio undefined, not infinite-but-reportable,
    so it is reported as a coverage fact instead)."""
    out = {}
    for jk in env.valid:
        ns = [len(env.results.get((jk, m), [])) for m in MOVES]
        mx, mn = max(ns), min(ns)
        out[jk] = {"n_by_operator": {m: n for m, n in zip(MOVES, ns)},
                   "max_n": mx, "min_n": mn,
                   "operators_covered": sum(1 for n in ns if n > 0),
                   "skew_max_over_min": round(mx / mn, 2) if mn else None}
    return out


def clears(summary):
    """Per judge: did the observed rank1-vs-rank2 gap beat that judge's own
    null-model p95 in this run? Returns {judge: bool/None}; None when the gap or
    the null model could not be formed (fewer than two operators with data, or
    fewer than four observations)."""
    out = {}
    for jk, g in (summary.get("final_gap_by_judge") or {}).items():
        nb = (summary.get("null_model_by_judge") or {}).get(jk)
        if not g or not nb or "gap_p95" not in nb:
            out[jk] = None
        else:
            out[jk] = bool(g["gap"] > nb["gap_p95"])
    return out


def run_one(policy_name, budget, seed):
    agent = POLICIES[policy_name]
    rounds = rounds_for(budget)
    bank = Bank()
    judges = load_judges()
    log_path = os.path.join(
        LOG_DIR, f"exploration_log_sweep_{policy_name}_b{budget}_seed{seed}.jsonl")
    env = Environment(judges, bank=bank, budget=budget, seed=seed, log_path=log_path)

    trace = agent(env, rounds=rounds)
    o = env.observe()

    summary = {
        "policy": policy_name,
        "budget": budget,
        "seed": seed,
        "rounds_allowed": rounds,
        "rounds_actually_run": len(trace),
        "stopped_reason": "budget_exhausted" if o["budget_left"] <= 0 else "rounds_exhausted",
        "steps_taken": env.step_id,
        "budget_left": o["budget_left"],
        "rejections": o["rejections"],
        "rejection_rate": round(o["rejections"] / env.step_id, 3) if env.step_id else None,
        "judges_valid": o["judges_valid"],
        "judges_excluded": o["judges_excluded"],
        "best_move_so_far": o["best_move_so_far"],
        "final_gap_by_judge": {jk: env._gap(jk) for jk in o["judges_valid"]},
        "null_model_by_judge": null_table(env),
        "allocation_by_judge": allocation_table(env),
        "cells": cell_table(env),
        "log_path": os.path.relpath(log_path, HERE),
        "log_lines": sum(1 for _ in open(log_path)),
    }
    summary["clears_null_p95_by_judge"] = clears(summary)
    return summary


def load_reference():
    """Read the committed budget=2000 runs back so the table has its reference
    row without re-running anything. Those files are inputs here, never outputs."""
    rows = []
    for policy, fname in REFERENCE_FILES.items():
        path = os.path.join(HERE, fname)
        for s in json.load(open(path)):
            rows.append({"policy": policy, "budget": s["budget"], "seed": s["seed"],
                         "source_file": fname,
                         "final_gap_by_judge": s["final_gap_by_judge"],
                         "null_model_by_judge": s["null_model_by_judge"],
                         "cells": s["cells"],
                         "clears_null_p95_by_judge": clears(s)})
    return rows


def alloc_from_cells(cells, judges):
    """Allocation spread recomputed from a committed run's `cells` block, which
    omits zero-n cells -- so a missing (judge, operator) key means n=0."""
    out = {}
    for jk in judges:
        ns = [cells.get(f"{jk}|{m}", {}).get("n", 0) for m in MOVES]
        mx, mn = max(ns), min(ns)
        out[jk] = {"max_n": mx, "min_n": mn,
                   "operators_covered": sum(1 for n in ns if n > 0),
                   "skew_max_over_min": round(mx / mn, 2) if mn else None}
    return out


def aggregate(rows):
    """Collapse the per-(policy, budget, seed) rows into one row per
    (policy, budget): cells cleared out of 9, mean gap, and allocation spread."""
    table = {}
    for r in rows:
        key = (r["policy"], r["budget"])
        b = table.setdefault(key, {"policy": r["policy"], "budget": r["budget"],
                                   "cells_total": 0, "cells_cleared": 0,
                                   "cells_undetermined": 0, "gaps": [], "p95s": [],
                                   "skews": [], "max_ns": [], "min_ns": [],
                                   "cells_missing_an_operator": 0})
        for jk, ok in r["clears_null_p95_by_judge"].items():
            b["cells_total"] += 1
            if ok is None:
                b["cells_undetermined"] += 1
            elif ok:
                b["cells_cleared"] += 1
        for g in (r["final_gap_by_judge"] or {}).values():
            if g:
                b["gaps"].append(g["gap"])
        for nb in (r["null_model_by_judge"] or {}).values():
            b["p95s"].append(nb["gap_p95"])
        alloc = r.get("allocation_by_judge") or alloc_from_cells(
            r["cells"], sorted(r["final_gap_by_judge"]))
        for a in alloc.values():
            b["max_ns"].append(a["max_n"])
            b["min_ns"].append(a["min_n"])
            if a["operators_covered"] < len(MOVES):
                b["cells_missing_an_operator"] += 1
            if a["skew_max_over_min"] is not None:
                b["skews"].append(a["skew_max_over_min"])

    out = []
    for key in sorted(table, key=lambda k: (k[1], k[0])):
        b = table[key]
        out.append({
            "policy": b["policy"],
            "budget": b["budget"],
            "cells_clearing_null_p95": f"{b['cells_cleared']}/{b['cells_total']}",
            "cells_undetermined": b["cells_undetermined"],
            "mean_gap": round(st.mean(b["gaps"]), 3) if b["gaps"] else None,
            "mean_null_p95": round(st.mean(b["p95s"]), 3) if b["p95s"] else None,
            "mean_max_n": round(st.mean(b["max_ns"]), 1) if b["max_ns"] else None,
            "mean_min_n": round(st.mean(b["min_ns"]), 1) if b["min_ns"] else None,
            "median_skew_max_over_min": (round(st.median(b["skews"]), 2)
                                         if len(b["skews"]) == b["cells_total"] else None),
            "skew_undefined_cells": b["cells_total"] - len(b["skews"]),
            "cells_missing_an_operator": b["cells_missing_an_operator"],
        })
    return out


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    runs = []
    for budget in BUDGETS:
        for policy in ("greedy", "random"):
            for seed in SEEDS:
                r = run_one(policy, budget, seed)
                runs.append(r)
                print(f"{policy:6} b={budget:<5} seed={seed:<3} steps={r['steps_taken']:<5} "
                      f"stopped={r['stopped_reason']:<17} "
                      f"clears={r['clears_null_p95_by_judge']}")

    reference = load_reference()
    table = aggregate(runs + reference)

    out = {
        "what": "greedy_agent vs random_agent across binding budgets",
        "space_size_probe_cells": 1458,
        "budgets_run_here": BUDGETS,
        "reference_budget_not_re_run": REFERENCE_BUDGET,
        "reference_sources": REFERENCE_FILES,
        "seeds": SEEDS,
        "comparison_table": table,
        "runs": runs,
    }
    out_path = os.path.join(HERE, "budget_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'policy':7} {'budget':>7} {'clears/9':>9} {'mean_gap':>9} {'mean_p95':>9} "
          f"{'max_n':>7} {'min_n':>7} {'skew':>8} {'gaps/9':>7}")
    for row in table:
        print(f"{row['policy']:7} {row['budget']:>7} {row['cells_clearing_null_p95']:>9} "
              f"{str(row['mean_gap']):>9} {str(row['mean_null_p95']):>9} "
              f"{str(row['mean_max_n']):>7} "
              f"{str(row['mean_min_n']):>7} {str(row['median_skew_max_over_min']):>8} "
              f"{row['cells_missing_an_operator']:>7}")
    print("gaps/9 = (judge, seed) cells in which at least one of the six operators "
          "was never probed")


if __name__ == "__main__":
    main()
