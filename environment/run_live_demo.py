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

MODEL SELECTION. The defaults are four non-Qwen models. Qwen is deliberately
avoided as a default judge because Qwen generated the rewrites in this
repository, so a Qwen judge would be scoring its own family's text -- preference
leakage, the same reason the frozen judge panel is all non-Qwen. You can still
pass one with --models; the caveat is yours to carry, and this script prints it.
One default is safety-tuned (gpt-oss-safeguard-20b), which is the interesting
case: the frozen data already found that safety tuning gave no measurable
protection against euphemism, and this is that claim re-asked live.

THE FOURTH DEFAULT IS A CHINESE-LAB MODEL, and it is here for a reason that is
scientific before it is diplomatic. Every one of the ten frozen judge families
is a Western lab -- AI2, Google, Meta, Microsoft, Mistral, OpenAI, TII. If the
euphemism effect is an artefact of one training tradition's data, safety
tuning, or refusal style, a lineage the environment was never built against is
where that would show. DeepSeek is not in the frozen data, was not consulted
while the bank or the prompt were designed, and is trained by a lab with no
connection to any of the ten. It is the cleanest available out-of-distribution
test of the finding, and the gate is applied to it on exactly the same terms.

PROVIDERS. A model id may carry an "@provider" suffix routing it to a different
endpoint in the same run (see LIVE_PROVIDERS in env.py). This is what lets one
command put a Groq-served model and a Chinese-lab model side by side under one
environment. It also matters for reproducibility from mainland China, where
huggingface.co is unreachable: the same DeepSeek model id can be re-run through
@modelscope or @deepseek by changing one word, with no other change anywhere.

Run:  python3 environment/run_live_demo.py
      python3 environment/run_live_demo.py --models openai/gpt-oss-120b,qwen/qwen3.6-27b
      python3 environment/run_live_demo.py --models zai-org/GLM-4.7-Flash@hf
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
    # OUT-OF-DISTRIBUTION LINEAGE. No Chinese lab appears in the ten frozen
    # judge families, and none was consulted while the bank or prompt were
    # built. Routed via @hf because Groq serves no non-Qwen Chinese model;
    # swap to @modelscope or @deepseek to reach it from mainland China.
    "deepseek-ai/DeepSeek-V3.2-Exp@hf",
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

    specs = [m.strip() for m in a.models.split(",") if m.strip()]
    # Resolve every route up front so an unknown provider name fails before a
    # single API call is spent, rather than half way through a paid sweep.
    routes = []
    for s in specs:
        model_id, prov_url, key_envvars = env.resolve_provider(s)
        # An explicit --base-url still wins, exactly as it did before routing existed.
        routes.append({"spec": s, "model": model_id,
                       "base_url": a.base_url or prov_url,
                       "key_envvars": key_envvars,
                       "provider": (s.partition("@")[2].strip().lower()
                                    or env.DEFAULT_LIVE_PROVIDER)})

    bank = env.Bank()
    print("=" * 78)
    print("ONE ENVIRONMENT, SEVERAL LIVE MODELS")
    print("=" * 78)
    print(f"  bank      : {len(bank)} acts x {len(env.MOVES)} operators (held constant)")
    print(f"  gate      : a model is probed only if its live literal mean <= {env.GATE}")
    print(f"  models    : {len(routes)}")
    for r in routes:
        endpoint = (r["base_url"] or os.environ.get("GOAI_LIVE_BASE_URL")
                    or env.DEFAULT_LIVE_BASE_URL)
        print(f"              - {r['model']:<34} via {r['provider']:<11} {endpoint}"
              + ("   [!] " + QWEN_CAVEAT if "qwen" in r["model"].lower() else ""))
    print()

    results = []
    for r in routes:
        m = r["model"]
        print("-" * 78)
        log_path = os.path.join(
            HERE, "exploration_log_live_" + m.replace("/", "_").replace(".", "-") + ".jsonl")
        try:
            out = env.live_demo(model=m, rounds=a.rounds, n=a.n, budget=a.budget,
                                seed=a.seed, calibration_n=a.calibration_n,
                                base_url=r["base_url"], cache_path=a.cache,
                                bank=bank, log_path=log_path,
                                min_interval=a.min_interval,
                                key_envvars=r["key_envvars"])
            out["error"] = None
        except env.LiveAPIError as e:
            # One model being unavailable must not kill the sweep -- report and move on.
            print(f"  live call failed for {m}: {e}")
            out = {"model": m, "judge": "live:" + m, "instrument_valid": None,
                   "literal_mean": None, "probes": [], "error": str(e)}
        out["provider"] = r["provider"]
        out["spec"] = r["spec"]
        if "qwen" in m.lower():
            out["caveat"] = QWEN_CAVEAT
        results.append(out)
        print()

    print("=" * 78)
    print("SUMMARY -- same environment, same prompt, same gate, different models")
    print("=" * 78)
    print(f"  {'model':<32} {'provider':<11} {'lit.mean':>9} {'valid':>6} {'probes':>7} "
          f"{'rank1':>17} {'flips':>6}")
    print("  " + "-" * 86)
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
        print(f"  {r['model'] if 'model' in r else r['judge']:<32} "
              f"{r.get('provider', '-'):<11} {lm:>9} {valid:>6} "
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
        "endpoints": {r["provider"]: (r["base_url"]
                                      or os.environ.get("GOAI_LIVE_BASE_URL")
                                      or env.DEFAULT_LIVE_BASE_URL)
                      for r in routes},
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
