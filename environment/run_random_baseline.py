#!/usr/bin/env python3
"""The random-exploration reference-frame baseline, run as a real campaign.

The GOAI rules require a "reference-frame design" (参照系设计) as one of the
three mandatory deliverables, and say explicitly that it "can be random
exploration, a trivial solution, or a simple baseline" (参照系可以是随机探索、
平凡解或简单基线). env.py already has a statistical baseline for this
(Environment.baseline_null_model, a shuffle of collected deltas), but that is
not an actual agent taking its own steps through the same loop greedy_agent
uses. random_agent (in env.py) is: same observe()/step() loop, uniformly
random (judge, operator) pick every step, no breadth/depth logic.

This script runs random_agent across the same 3 seeds and the same budget as
run_campaign.py's greedy_agent campaign, so the two are directly comparable.
It reuses run_campaign.py's judge list / budget / rounds / seed config and its
load_judges() setup rather than duplicating any of that, and imports Bank /
Environment / random_agent / MOVES from env.py -- nothing about the fixed
setup is re-implemented here.

Output goes to its own files, distinct from the greedy_agent campaign's:
  exploration_log_random_seed{0,1,42}.jsonl   (same JSONL schema as the
                                                greedy campaign logs)
  random_baseline_summary.json

Nothing here writes to exploration_log_campaign_seed*.jsonl or
campaign_summary.json -- those are the official greedy_agent results.

Usage: python3 run_random_baseline.py
"""
import json
import os
import statistics as st

from env import Bank, Environment, random_agent, MOVES
from run_campaign import load_judges, BUDGET, ROUNDS, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))


def run_seed(seed):
    bank = Bank()
    judges = load_judges()
    log_path = os.path.join(HERE, f"exploration_log_random_seed{seed}.jsonl")
    env = Environment(judges, bank=bank, budget=BUDGET, seed=seed, log_path=log_path)

    trace = random_agent(env, rounds=ROUNDS)

    o = env.observe()

    cells = {}
    for jk in env.valid:
        for m in MOVES:
            v = env.results.get((jk, m), [])
            if v:
                cells[f"{jk}|{m}"] = {"n": len(v), "mean_delta": round(st.mean(v), 3),
                                       "stdev": round(st.pstdev(v), 3) if len(v) > 1 else 0.0}

    null_by_judge = {}
    for jk in env.valid:
        nb = env.baseline_null_model(jk)
        if "gap_p95" in nb:
            null_by_judge[jk] = nb

    summary = {
        "seed": seed,
        "policy": "random_agent",
        "budget": BUDGET,
        "rounds_allowed": ROUNDS,
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
        # Same null model run_campaign.py applies to the greedy policy, on the
        # same code path. Running it here is the whole point of the comparison:
        # the shuffle conditions on the REALIZED per-operator sample sizes, so a
        # policy that concentrates its budget on one operator raises its own
        # significance bar. Without this the two policies cannot be compared on
        # anything except raw gap, which is the comparison that misleads.
        "null_model_by_judge": null_by_judge,
        "cells": cells,
        # Recorded relative to this file, not absolute: an absolute path is
        # specific to the machine that ran it and is not reproducible for anyone else.
        "log_path": os.path.relpath(log_path, HERE),
        "log_lines": sum(1 for _ in open(log_path)),
    }
    return summary


def main():
    summaries = [run_seed(s) for s in SEEDS]
    out_path = os.path.join(HERE, "random_baseline_summary.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out_path}")
    for s in summaries:
        print(f"seed={s['seed']}: steps={s['steps_taken']} stopped={s['stopped_reason']} "
              f"rejections={s['rejections']} ({s['rejection_rate']}) "
              f"gaps={s['final_gap_by_judge']}")


if __name__ == "__main__":
    main()
