#!/usr/bin/env python3
"""The UCB1 policy, run as a real campaign on the same terms as the other two.

WHY A THIRD POLICY. The submission had exactly two: greedy_agent (breadth then
depth, concentrating budget on the current leader) and random_agent (uniform,
the reference frame the GOAI rules require). At the official budget, greedy
clears its own null-model p95 in 0 of 9 (judge, seed) cells while random clears
6 of 9, because greedy's depth phase drives five of six operators to n=3-26
while one gets n=364-459, and baseline_null_model's shuffle conditions on those
realized sample sizes -- so greedy raises its own significance bar to 1.4-3.1
while random's sits near 0.4-0.5.

That result is honest but ambiguous. A reviewer's fair question is whether it
says something about exploration in this environment or only about one badly
designed greedy policy. UCB1 is the standard principled way to trade
exploration against exploitation, so running it here is what separates those
two readings. If a principled policy also fails to beat random, that is a
finding about the environment; if it lands between the two, the environment is
demonstrably sensitive to where a policy sits on that tradeoff.

Same terms as the other two campaigns, so the three are comparable and nothing
about the comparison is an artefact of a differently configured run: the judge
list, budget, rounds and seeds are IMPORTED from run_campaign.py rather than
restated, the setup comes from its load_judges(), and Bank / Environment /
ucb_agent / MOVES come from env.py. ucb_agent pulls n=6 acts per round, exactly
as greedy_agent and random_agent do.

Output goes to its own files:
  exploration_log_ucb_seed{0,1,42}.jsonl   (same JSONL schema as the greedy and
                                            random campaign logs)
  ucb_summary.json

Nothing here writes to exploration_log_campaign_seed*.jsonl,
campaign_summary.json, exploration_log_random_seed*.jsonl or
random_baseline_summary.json -- those are the existing official results.

Usage: python3 run_ucb_baseline.py
"""
import json
import os
import statistics as st

from env import Bank, Environment, ucb_agent, MOVES
from run_campaign import load_judges, BUDGET, ROUNDS, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))

# Textbook UCB1. Kept as a named constant rather than left implicit because the
# reward normalization in ucb_agent compresses reward differences relative to
# the exploration bonus, which makes c the parameter that actually decides where
# this policy sits between greedy and random. A reader comparing the three
# policies needs to see the value that produced these numbers.
UCB_C = 1.0


def run_seed(seed):
    bank = Bank()
    judges = load_judges()
    log_path = os.path.join(HERE, f"exploration_log_ucb_seed{seed}.jsonl")
    env = Environment(judges, bank=bank, budget=BUDGET, seed=seed, log_path=log_path)

    trace = ucb_agent(env, rounds=ROUNDS, c=UCB_C)

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
        "policy": "ucb_agent",
        "ucb_c": UCB_C,
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
        # The same null model run_campaign.py and run_random_baseline.py apply,
        # on the same code path. It is the entire basis of the comparison: the
        # shuffle conditions on the REALIZED per-operator sample sizes, so each
        # policy is measured against the bar its own allocation created. Without
        # it the three policies can only be compared on raw gap, which is the
        # comparison that misleads.
        "null_model_by_judge": null_by_judge,
        "cells": cells,
        "log_path": log_path,
        "log_lines": sum(1 for _ in open(log_path)),
    }
    return summary


def main():
    summaries = [run_seed(s) for s in SEEDS]
    out_path = os.path.join(HERE, "ucb_summary.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out_path}")
    for s in summaries:
        print(f"seed={s['seed']}: steps={s['steps_taken']} stopped={s['stopped_reason']} "
              f"rejections={s['rejections']} ({s['rejection_rate']}) "
              f"gaps={s['final_gap_by_judge']}")


if __name__ == "__main__":
    main()
