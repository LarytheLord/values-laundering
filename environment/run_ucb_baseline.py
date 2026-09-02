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

THE EXPLORATION CONSTANT IS A FLAG, NOT A LITERAL. c used to be a module
constant with no way to override it, which meant the c-sweep reported in the
write-up could not be re-derived from this repository at all: the only
committed artifact was ucb_summary.json, three runs all at c=1.0. --c fixes
that. The default is still 1.0, so a bare `python3 run_ucb_baseline.py`
reproduces the committed artifact exactly and byte-for-byte.

Any c other than the default writes its per-step logs under sweep_logs/ with
the c in the filename, and a non-default c with no explicit --out refuses to
run rather than silently overwriting ucb_summary.json. Both guards exist for
the same reason: the committed c=1.0 artifacts are cited in the write-up, and
a swept run must not be able to clobber them by accident.

Usage: python3 run_ucb_baseline.py                        (the committed run)
       python3 run_ucb_baseline.py --c 0.0 --out /tmp/ucb_c0.json
See run_ucb_c_sweep.py for the full sweep over c that produces
ucb_c_sweep_summary.json.
"""
import argparse
import json
import os
import statistics as st

from env import Bank, Environment, ucb_agent, MOVES
from run_campaign import load_judges, BUDGET, ROUNDS, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_LOG_DIR = os.path.join(HERE, "sweep_logs")

# Textbook UCB1. Kept as a named constant rather than left implicit because the
# reward normalization in ucb_agent compresses reward differences relative to
# the exploration bonus, which makes c the parameter that actually decides where
# this policy sits between greedy and random. A reader comparing the three
# policies needs to see the value that produced these numbers.
UCB_C = 1.0

# The committed artifact this script writes at the default c.
DEFAULT_OUT = os.path.join(HERE, "ucb_summary.json")


def c_tag(c):
    """A filename-safe rendering of c: 0.0 -> c0, 0.25 -> c0p25, 1.0 -> c1.
    The decimal point becomes 'p' so the tag never looks like a file
    extension to a glob or a shell."""
    return "c" + f"{c:g}".replace(".", "p").replace("-", "neg")


def default_log_path(seed, c):
    """Where this seed's per-step JSONL goes when the caller does not say.

    Only the default c writes to the committed exploration_log_ucb_seed*.jsonl
    path. Environment truncates its log file on construction, so a run at some
    other c that shared those filenames would overwrite a committed artifact
    with a different policy's trace. Routing every non-default c into
    sweep_logs/ makes that impossible rather than merely unlikely."""
    if c == UCB_C:
        return os.path.join(HERE, f"exploration_log_ucb_seed{seed}.jsonl")
    return os.path.join(SWEEP_LOG_DIR,
                        f"exploration_log_ucb_{c_tag(c)}_seed{seed}.jsonl")


def run_seed(seed, c=UCB_C, log_path=None):
    bank = Bank()
    judges = load_judges()
    log_path = log_path or default_log_path(seed, c)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = Environment(judges, bank=bank, budget=BUDGET, seed=seed, log_path=log_path)

    trace = ucb_agent(env, rounds=ROUNDS, c=c)

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
        "ucb_c": c,
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
        # Recorded relative to this file, not absolute: an absolute path is
        # specific to the machine that ran it and is not reproducible for anyone else.
        "log_path": os.path.relpath(log_path, HERE),
        "log_lines": sum(1 for _ in open(log_path)),
    }
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the UCB1 policy over the frozen replay environment.")
    p.add_argument("--c", type=float, default=UCB_C,
                   help=f"UCB1 exploration constant (default {UCB_C}, textbook "
                        "UCB1). c=0 turns the exploration bonus off entirely "
                        "and reduces the policy to pure exploitation.")
    p.add_argument("--out", default=None,
                   help="where to write the summary JSON (default "
                        "ucb_summary.json, and only permitted at the default c)")
    a = p.parse_args(argv)
    if a.c < 0:
        p.error("--c must be non-negative; a negative exploration constant "
                "would penalise under-sampled arms, which is not UCB1")
    if a.out is None:
        if a.c != UCB_C:
            # Refusing beats overwriting. ucb_summary.json is the committed
            # c=1.0 artifact the write-up cites, and a swept run landing on it
            # would replace a cited result with a different policy's numbers
            # while leaving the filename saying otherwise.
            p.error(f"--out is required when --c is not {UCB_C}: writing a "
                    f"c={a.c:g} run to the default ucb_summary.json would "
                    "overwrite the committed c=1.0 artifact")
        a.out = DEFAULT_OUT
    return a


def main(argv=None):
    args = parse_args(argv)
    summaries = [run_seed(s, c=args.c) for s in SEEDS]
    out_path = os.path.abspath(args.out)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out_path}")
    for s in summaries:
        print(f"seed={s['seed']}: c={s['ucb_c']:g} steps={s['steps_taken']} "
              f"stopped={s['stopped_reason']} "
              f"rejections={s['rejections']} ({s['rejection_rate']}) "
              f"gaps={s['final_gap_by_judge']}")
    return summaries


if __name__ == "__main__":
    main()
