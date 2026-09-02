#!/usr/bin/env python3
"""Does live mode agree with replay mode? Measure it, and report whichever way it comes out.

WHY THIS SCRIPT IS THE POINT. The rest of this repository replays frozen
judgments. That buys reproducibility, but it rests on an assumption a reviewer
has every right to challenge: that a replayed score is the score the model would
give if you actually asked it. This script tests that assumption instead of
asserting it. It picks a sample of (act, operator) cells, scores them live
through the same prompt that produced the frozen data, and compares the two
numbers cell by cell.

WHY openai/gpt-oss-120b. It is the one judge that is both in the frozen data and
still served by a live API today, so this is a SAME-MODEL comparison, not a
proxy. data/exp8_groq_results.json was itself produced by calling Groq with this
model id through analysis/exp8_groq_judges.py, using the identical NUM prompt now
frozen in env.LIVE_NUM_PROMPT. The frozen judges behind the headline campaign
(gemma-4, Olmo-3, granite-4.1) are Kaggle-local weights with no hosted endpoint,
so they cannot be re-asked this way -- which is exactly why this check runs on
the judge that can be.

WHAT WOULD FALSIFY REPLAY. Divergence. If live and replay disagree badly, replay
fidelity is weaker than the write-up assumes, and that is a real finding about
this environment that belongs in the report. This script is written so that
outcome is reported as plainly as agreement would be: it prints the numbers, it
does not grade them, and it never retries a sample to get a nicer answer.

KNOWN SOURCES OF LEGITIMATE DISAGREEMENT, stated in advance so they are not
mistaken for evidence either way:
  * The frozen exp8 run used max_tokens=8. openai/gpt-oss-120b as served today
    is a reasoning model and returns an EMPTY string at 8 tokens, so live mode
    must use a larger budget (512). Same prompt, same model id, different token
    budget, because the smaller one no longer works at all.
  * Providers re-quantize and re-serve a model id over time. "openai/gpt-oss-120b
    on Groq in August" and "openai/gpt-oss-120b on Groq today" are the same id,
    not provably the same bytes.
  * These judges score on a coarse 0-10 integer scale, so a one-point move is a
    large fraction of the observed range.

Run:  python3 environment/compare_live_replay.py
      python3 environment/compare_live_replay.py --n 20 --cache live_cache.json
"""
import argparse
import importlib.util
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location("env_live", os.path.join(HERE, "env.py"))
env = importlib.util.module_from_spec(_spec)
sys.modules["env_live"] = env
_spec.loader.exec_module(env)

# The frozen counterpart. Same model id, same prompt, recorded via the Groq API.
FROZEN_FILE = "exp8_groq_results.json"
FROZEN_MODEL = "openai/gpt-oss-120b"


class Exp8ReplayJudge:
    """ReplayJudge over data/exp8_groq_results.json instead of recency_results.json.

    Written as a separate adapter rather than by widening ReplayJudge with a
    second schema, for one reason: ReplayJudge is on the path that produced every
    official artifact in this directory, and those artifacts must stay
    byte-identical. Nothing here touches it.

    It exists at all because exp8 stores a different shape -- a flat
    {model: {act_text: {act, base, moves:{op:{score,delta}}}}} with no per-family
    wrapper and no binary fields -- and because implementing the judge interface
    a second time, against a second schema, is itself the evidence that
    .key/.version/.literal_mean/.instrument_valid/.covers/.score is a real seam
    and not a shape that only ReplayJudge happens to have.
    """

    def __init__(self, path=None, model=FROZEN_MODEL):
        path = path or env.find_data(FROZEN_FILE)
        d = json.load(open(path))
        if model not in d:
            raise KeyError(f"{model} not in {FROZEN_FILE}; have {sorted(d)}")
        self.key = "replay:" + model
        self.version = model
        self._acts = d[model]
        bases = [v["base"] for v in self._acts.values() if v.get("base") is not None]
        self.literal_mean = st.mean(bases) if bases else None
        self.instrument_valid = (self.literal_mean is not None
                                 and self.literal_mean <= env.GATE)

    def covers(self, act_text, move):
        r = self._acts.get(act_text)
        return bool(r and r.get("moves", {}).get(move) is not None)

    def score(self, act_text, move):
        r = self._acts.get(act_text)
        if not r:
            return None
        mv = r.get("moves", {}).get(move)
        if mv is None:
            return None
        base, rewritten = r.get("base"), mv.get("score")
        flip = (None if base is None or rewritten is None
                else (base <= env.GATE and rewritten > env.GATE))
        return {"literal": base, "rewritten": rewritten, "delta": mv.get("delta"),
                "binary_pair": (None, None), "threshold_flip": flip,
                "reasoning": None}


def pearson(xs, ys):
    """Pearson r, stdlib only. None when it is undefined rather than 0.0.

    Undefined here is not a corner case to paper over: if every live score in the
    sample is identical, the variance is zero and a correlation genuinely does
    not exist. Returning None says that; returning 0.0 would claim 'no
    relationship', which is a different and false statement.
    """
    n = len(xs)
    if n < 2:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def agree(name, live, rep):
    """Paired agreement stats for one quantity."""
    pairs = [(a, b) for a, b in zip(live, rep) if a is not None and b is not None]
    if not pairs:
        return {"quantity": name, "n": 0, "error": "no comparable pairs"}
    lv = [p[0] for p in pairs]
    rv = [p[1] for p in pairs]
    diffs = [a - b for a, b in pairs]
    r = pearson(lv, rv)
    return {
        "quantity": name,
        "n": len(pairs),
        "pearson_r": None if r is None else round(r, 4),
        "mean_abs_diff": round(st.mean([abs(d) for d in diffs]), 4),
        "mean_signed_diff_live_minus_replay": round(st.mean(diffs), 4),
        "max_abs_diff": max(abs(d) for d in diffs),
        "exact_match_rate": round(sum(1 for d in diffs if d == 0) / len(diffs), 4),
        "within_1_rate": round(sum(1 for d in diffs if abs(d) <= 1) / len(diffs), 4),
        "live_mean": round(st.mean(lv), 4),
        "replay_mean": round(st.mean(rv), 4),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=20,
                   help="number of (act, operator) cells to compare (default 20)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=FROZEN_MODEL,
                   help="model id, must be present in the frozen file AND live")
    p.add_argument("--base-url", default=None)
    p.add_argument("--calibration-n", type=int, default=8)
    p.add_argument("--min-interval", type=float, default=2.0)
    p.add_argument("--cache", default=None,
                   help="cache raw API responses here; makes a rerun free and "
                        "resume-safe (default: no cache)")
    p.add_argument("--out", default=os.path.join(HERE, "live_vs_replay_report.json"),
                   help="NEW output file; never an official artifact")
    a = p.parse_args()

    bank = env.Bank()
    frozen = Exp8ReplayJudge(model=a.model)
    print(f"Frozen  : {frozen.key}  literal_mean={frozen.literal_mean:.3f}  "
          f"instrument_valid={frozen.instrument_valid}")

    # Only cells BOTH sides can answer: the bank must have the rewrite text (live
    # needs it to ask) and the frozen file must have a recorded score.
    cells = [(x["act"], m) for x in bank.acts for m in env.MOVES
             if frozen.covers(x["act"], m)]
    rng = random.Random(a.seed)
    rng.shuffle(cells)
    sample = cells[:a.n]
    print(f"Cells   : {len(cells)} comparable, sampling {len(sample)} with seed {a.seed}\n")

    live = env.LiveJudge(model=a.model, bank=bank, calibration_n=a.calibration_n,
                         seed=a.seed, base_url=a.base_url, cache_path=a.cache,
                         min_interval=a.min_interval)
    lm = "n/a" if live.literal_mean is None else f"{live.literal_mean:.3f}"
    print(f"Live    : {live.key}  literal_mean={lm}  "
          f"instrument_valid={live.instrument_valid}")
    print(f"Gate    : both sides judged by the same rule, literal mean <= {env.GATE}")
    print(f"Verdict : the gate {'AGREES' if live.instrument_valid == frozen.instrument_valid else 'DISAGREES'} "
          f"across modes for this judge\n")

    hdr = (f"  {'#':>2}  {'operator':<17} {'lit L/R':>9} {'rew L/R':>9} "
           f"{'delta L/R':>11}   act")
    print(hdr)
    print("  " + "-" * 96)

    rows = []
    for i, (act, move) in enumerate(sample, 1):
        L = live.score(act, move)
        R = frozen.score(act, move)
        if L is None or R is None:
            continue
        rows.append({"act": act, "operator": move,
                     "live": {k: L[k] for k in ("literal", "rewritten", "delta",
                                                "threshold_flip")},
                     "replay": {k: R[k] for k in ("literal", "rewritten", "delta",
                                                  "threshold_flip")}})
        short = act if len(act) <= 44 else act[:43] + "…"
        f = lambda x: "--" if x is None else f"{x}"
        print(f"  {i:>2}  {move:<17} "
              f"{f(L['literal'])+'/'+f(R['literal']):>9} "
              f"{f(L['rewritten'])+'/'+f(R['rewritten']):>9} "
              f"{f(L['delta'])+'/'+f(R['delta']):>11}   {short}")

    lit = agree("literal_score", [r["live"]["literal"] for r in rows],
                [r["replay"]["literal"] for r in rows])
    rew = agree("rewritten_score", [r["live"]["rewritten"] for r in rows],
                [r["replay"]["rewritten"] for r in rows])
    dlt = agree("delta", [r["live"]["delta"] for r in rows],
                [r["replay"]["delta"] for r in rows])

    flips = [(r["live"]["threshold_flip"], r["replay"]["threshold_flip"]) for r in rows]
    flips = [(x, y) for x, y in flips if x is not None and y is not None]
    flip_agree = (round(sum(1 for x, y in flips if x == y) / len(flips), 4)
                  if flips else None)

    print("\n" + "=" * 98)
    print("AGREEMENT (live vs replay, same model id, same prompt)")
    print("=" * 98)
    for s in (lit, rew, dlt):
        if s.get("n"):
            r = "n/a" if s["pearson_r"] is None else f"{s['pearson_r']:+.4f}"
            print(f"  {s['quantity']:<16} n={s['n']:<3} r={r:>8}  "
                  f"MAE={s['mean_abs_diff']:.3f}  bias={s['mean_signed_diff_live_minus_replay']:+.3f}  "
                  f"exact={s['exact_match_rate']:.0%}  within1={s['within_1_rate']:.0%}")
        else:
            print(f"  {s['quantity']:<16} {s.get('error')}")
    if flip_agree is not None:
        print(f"  {'threshold_flip':<16} n={len(flips):<3} agreement={flip_agree:.0%}")

    report = {
        "_purpose": ("Live-vs-replay agreement for the values-laundering environment. "
                     "Answers the reproducibility question directly: if you re-ask the "
                     "model instead of replaying it, do you get the same number?"),
        "generated_by": "environment/compare_live_replay.py",
        "is_official_artifact": False,
        "note_on_artifacts": ("This file is NEW output. It does not replace or modify "
                              "campaign_summary.json, random_baseline_summary.json, "
                              "ucb_summary.json, or any exploration_log*.jsonl."),
        "model": a.model,
        "same_model_both_sides": True,
        "frozen_source": f"data/{FROZEN_FILE}",
        "frozen_generated_by": "analysis/exp8_groq_judges.py (Groq API, max_tokens=8)",
        "live_endpoint": getattr(live.client, "base_url", None),
        "live_max_tokens": getattr(live.client, "max_tokens", None),
        "prompt": env.LIVE_NUM_PROMPT,
        "prompt_provenance": ("byte-identical to the NUM constant in the kernels that "
                              "generated both data/recency_results.json and "
                              "data/exp8_groq_results.json"),
        "n_requested": a.n,
        "n_compared": len(rows),
        "seed": a.seed,
        "gate": env.GATE,
        "instrument_validity": {
            "live": {"literal_mean": live.literal_mean,
                     "instrument_valid": live.instrument_valid,
                     "n_calibration_acts": a.calibration_n},
            "replay": {"literal_mean": frozen.literal_mean,
                       "instrument_valid": frozen.instrument_valid,
                       "n_acts": len(frozen._acts)},
            "gate_decision_agrees": live.instrument_valid == frozen.instrument_valid,
        },
        "agreement": {"literal_score": lit, "rewritten_score": rew, "delta": dlt,
                      "threshold_flip_agreement": flip_agree,
                      "threshold_flip_n": len(flips)},
        "known_legitimate_sources_of_disagreement": [
            "frozen run used max_tokens=8; today the same model id is served as a "
            "reasoning model and returns an empty string at 8 tokens, so live uses 512",
            "a provider may re-quantize or re-serve a model id over time; same id is "
            "not provably the same weights",
            "0-10 integer scale, so one point is a large fraction of the observed range",
        ],
        "api_accounting": live.stats(),
        "cells": rows,
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  API accounting: {live.stats()}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    try:
        main()
    except env.LiveAPIError as e:
        raise SystemExit(f"\nlive mode could not run:\n  {e}\n")
