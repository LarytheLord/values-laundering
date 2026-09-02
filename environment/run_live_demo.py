#!/usr/bin/env python3
"""Run the SAME environment against SEVERAL live models, and show it does not care which.

The organisers' guidance was explicit on two points: the submitted code must be
reproducible against a live API, and -- "甚至建议大家试几个不同模型的 API，看看你提出
的这套环境和架构对不同大模型是不是都 work" -- the environment should be shown to work
across different large models, not just one. `env.py --live` answers the first.
This script answers the second.

It builds one LiveJudge per model, hands each to the same Environment class the
official replay campaigns use, runs the same breadth-then-depth loop, and prints
the per-model result side by side. Nothing model-specific exists anywhere in the
environment: a model id is a string, the judge interface is the only contract,
and the validity gate is applied to every model on identical terms.

WHAT TO WATCH FOR, and it is not "all models agree". They should not all agree,
and the interesting output is where they differ:
  * whether each model clears the instrument-validity gate at all (a model that
    rates the literal cruelty acts as acceptable is not a usable instrument, and
    the environment excludes it rather than reporting its numbers);
  * whether the rank-1 operator is stable across models;
  * whether any probe produces a threshold flip.

MODEL SELECTION. The defaults are three non-Qwen models. Qwen is deliberately
avoided as a default judge because Qwen generated the rewrites in this
repository, so a Qwen judge would be scoring its own family's text -- preference
leakage, the same reason the frozen judge panel is all non-Qwen. You can still
pass one with --models; the caveat is yours to carry, and this script prints it.
One default is safety-tuned (gpt-oss-safeguard-20b), which is the interesting
case: the frozen data already found that safety tuning gave no measurable
protection against euphemism, and this is that claim re-asked live.

Run:  python3 environment/run_live_demo.py
      python3 environment/run_live_demo.py --models openai/gpt-oss-120b,qwen/qwen3.6-27b
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("env_live_demo", os.path.join(HERE, "env.py"))
env = importlib.util.module_from_spec(_spec)
sys.modules["env_live_demo"] = env
_spec.loader.exec_module(env)

DEFAULT_MODELS = [
    "openai/gpt-oss-120b",           # also present in the frozen data -- comparable
    "openai/gpt-oss-20b",            # same family, ~6x smaller: does size change it?
    "openai/gpt-oss-safeguard-20b",  # SAFETY-TUNED: does that buy any protection?
]

QWEN_CAVEAT = ("this model is from the Qwen family, which GENERATED the rewrites in "
               "this repository -- treat its scores as preference-leaking, not as an "
               "independent judge")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma-separated model ids (default: %(default)s)")
    p.add_argument("--budget", type=int, default=6,
                   help="acts scored per model (default 6; keep small, this is a demo)")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--n", type=int, default=1, help="acts per decision (default 1)")
    p.add_argument("--calibration-n", type=int, default=4,
                   help="literal acts scored live per model to test the gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default=None)
    p.add_argument("--min-interval", type=float, default=2.0)
    p.add_argument("--cache", default=None,
                   help="cache raw API responses here so a rerun is free")
    p.add_argument("--out", default=os.path.join(HERE, "live_multimodel_report.json"),
                   help="NEW output file; never an official artifact")
    a = p.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    bank = env.Bank()
    print("=" * 78)
    print("ONE ENVIRONMENT, SEVERAL LIVE MODELS")
    print("=" * 78)
    print(f"  bank      : {len(bank)} acts x {len(env.MOVES)} operators (held constant)")
    print(f"  endpoint  : {a.base_url or os.environ.get('GOAI_LIVE_BASE_URL') or env.DEFAULT_LIVE_BASE_URL}")
    print(f"  gate      : a model is probed only if its live literal mean <= {env.GATE}")
    print(f"  models    : {len(models)}")
    for m in models:
        print(f"              - {m}" + ("   [!] " + QWEN_CAVEAT if "qwen" in m.lower() else ""))
    print()

    results = []
    for m in models:
        print("-" * 78)
        log_path = os.path.join(
            HERE, "exploration_log_live_" + m.replace("/", "_").replace(".", "-") + ".jsonl")
        try:
            out = env.live_demo(model=m, rounds=a.rounds, n=a.n, budget=a.budget,
                                seed=a.seed, calibration_n=a.calibration_n,
                                base_url=a.base_url, cache_path=a.cache,
                                bank=bank, log_path=log_path,
                                min_interval=a.min_interval)
            out["error"] = None
        except env.LiveAPIError as e:
            # One model being unavailable must not kill the sweep -- report and move on.
            print(f"  live call failed for {m}: {e}")
            out = {"model": m, "judge": "live:" + m, "instrument_valid": None,
                   "literal_mean": None, "probes": [], "error": str(e)}
        if "qwen" in m.lower():
            out["caveat"] = QWEN_CAVEAT
        results.append(out)
        print()

    print("=" * 78)
    print("SUMMARY -- same environment, same prompt, same gate, different models")
    print("=" * 78)
    print(f"  {'model':<32} {'lit.mean':>9} {'valid':>6} {'probes':>7} "
          f"{'rank1':>17} {'flips':>6}")
    print("  " + "-" * 74)
    for r in results:
        lm = "n/a" if r.get("literal_mean") is None else f"{r['literal_mean']:.2f}"
        probes = r.get("probes") or []
        best, flips = "-", 0
        if probes:
            per = {}
            for pr in probes:
                per.setdefault(pr["operator"], []).append(pr["delta"])
            per = {k: sum(v) / len(v) for k, v in per.items()}
            best = max(per, key=per.__getitem__)
            flips = sum(1 for pr in probes if pr["threshold_flip"])
        valid = ("--" if r.get("instrument_valid") is None
                 else ("yes" if r["instrument_valid"] else "NO"))
        print(f"  {r['model'] if 'model' in r else r['judge']:<32} {lm:>9} {valid:>6} "
              f"{len(probes):>7} {best:>17} {flips:>6}")

    ok = [r for r in results if r.get("instrument_valid")]
    print()
    if not ok:
        print("  No model cleared the gate in this run. With a budget this small that is "
              "a statement about the sample size, not about the models.")
    else:
        print(f"  {len(ok)}/{len(results)} models cleared the validity gate and were probed "
              f"by the same loop, with no environment change of any kind.")
        print("  Per-model rank-1 above is from a deliberately tiny budget and is NOT a "
              "finding; the campaign runs are what carry statistical weight.")

    report = {
        "_purpose": ("Portability check: the same environment, prompt and gate run "
                     "against several live models via an OpenAI-compatible API."),
        "generated_by": "environment/run_live_demo.py",
        "is_official_artifact": False,
        "note_on_artifacts": ("NEW output. Does not replace or modify "
                              "campaign_summary.json, random_baseline_summary.json, "
                              "ucb_summary.json, or any exploration_log*.jsonl "
                              "produced by the replay campaigns."),
        "endpoint": a.base_url or os.environ.get("GOAI_LIVE_BASE_URL") or env.DEFAULT_LIVE_BASE_URL,
        "prompt": env.LIVE_NUM_PROMPT,
        "gate": env.GATE,
        "budget_per_model": a.budget,
        "calibration_n": a.calibration_n,
        "seed": a.seed,
        "caveat_on_rank1": ("budget per model here is tiny by design; per-model rank-1 "
                            "is an illustration that the loop runs, not a result"),
        "models": results,
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
