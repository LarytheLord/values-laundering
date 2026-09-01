#!/usr/bin/env python3
"""A real (non-selftest) exploration campaign against the environment in env.py.

selftest() runs one seed, budget=300, rounds=14 -> stops after 84 steps with 216
of the 300 budget unused (rounds is the binding constraint, not budget). This
script removes that artificial cap: it runs the SAME greedy_agent policy against
the SAME replay environment, but with a budget large enough to actually bind,
and across multiple seeds, so the campaign produces a real signal about the
policy rather than a one-shot smoke test.

Nothing about the environment's fixed part, feedback schema, or log format is
changed. This only calls the public API (Environment, ReplayJudge, Bank,
greedy_agent) with different parameters and writes each seed's trace to its own
log file, so the original 84-record exploration_log.jsonl smoke-test baseline
is left untouched.

Usage: python3 run_campaign.py
Output: exploration_log_campaign_seed{N}.jsonl per seed (immutable JSONL, same
        schema as exploration_log.jsonl), plus campaign_summary.json.
"""
import json
import os
import statistics as st

from env import Bank, ReplayJudge, Environment, greedy_agent, MOVES, GATE

HERE = os.path.dirname(os.path.abspath(__file__))

JUDGE_KEYS = ["gemma4-12b", "olmo3-7b", "gemma4-e4b", "granite-guard", "granite41-8b"]
BUDGET = 2000       # vs. 300 in selftest
ROUNDS = 350         # enough that BUDGET (not ROUNDS) is what stops the agent
SEEDS = [0, 1, 42]


def load_judges():
    judges = []
    for k in JUDGE_KEYS:
        try:
            judges.append(ReplayJudge(k))
        except KeyError as e:
            print(f"  skip {k}: {e}")
    return judges


def run_seed(seed):
    bank = Bank()
    judges = load_judges()
    log_path = os.path.join(HERE, f"exploration_log_campaign_seed{seed}.jsonl")
    env = Environment(judges, bank=bank, budget=BUDGET, seed=seed, log_path=log_path)

    trace = greedy_agent(env, rounds=ROUNDS)

    o = env.observe()

    # Per-(judge, move) cell stats as actually accumulated in this run --
    # includes both the breadth pass (all cells, small n) and whatever the
    # depth pass added (round-robin across judges_valid, so every judge's
    # leading operator gets deepened in turn, not just one judge's).
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
        "cells": cells,
        "null_model_by_judge": null_by_judge,
        # Recorded relative to this file, not absolute: an absolute path is
        # specific to the machine that ran it and is not reproducible for anyone else.
        "log_path": os.path.relpath(log_path, HERE),
        "log_lines": sum(1 for _ in open(log_path)),
    }
    return summary


def main():
    summaries = [run_seed(s) for s in SEEDS]
    out_path = os.path.join(HERE, "campaign_summary.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out_path}")
    for s in summaries:
        print(f"seed={s['seed']}: steps={s['steps_taken']} stopped={s['stopped_reason']} "
              f"rejections={s['rejections']} ({s['rejection_rate']}) "
              f"gaps={s['final_gap_by_judge']}")


if __name__ == "__main__":
    main()
