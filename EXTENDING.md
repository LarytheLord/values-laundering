# Extending this environment

The judging criteria for this track ask whether results can be consolidated into a problem package
or environment package someone else can continue. This file is that handover. It assumes you have
cloned the repository at tag `v1.0-goai-semifinal` and run `bash environment/scripts/smoke_test.sh`
once, which needs no install, no API key and no network.

## The three seams

Almost every extension is one of these three, and each is a small, local change.

### 1. Swap the act bank, keep everything else

`data/kernel_payload.json` is the only file that defines *what* is being judged. Its shape is:

```json
{"exp1": [{"act": "<literal description>",
           "moves": {"euphemism": "<rewrite>", "agent_deletion": "<rewrite>", ...}}]}
```

Point `Bank(path=...)` at a different file with the same shape and the entire environment,
both gates, all three policies, the null model and every analysis script work unchanged. This is
the seam for **cross-domain extension**, which is the single most valuable open direction: the
current bank is single-domain, and the report says so.

`environment/tests/test_env.py` builds synthetic banks this way in `tmp_path`, so there are
working examples of the format in the test suite.

### 2. Swap the judge, keep the bank

`ReplayJudge` and `LiveJudge` expose the same public surface, and `TestLiveJudgeIsDropInForReplayJudge`
asserts structurally that they stay interchangeable. To add a judge:

- **A hosted model:** add an entry to `LIVE_PROVIDERS` in `environment/env.py` with its
  `base_url` and the environment variables holding its key, then run
  `--models "<model-id>@<provider>"`. Nothing else changes.
- **Local weights, or a non-OpenAI-compatible API:** write a callable that takes one prompt string
  and returns one reply string, and pass it as `client=`. `FakeClient` in the test suite is a
  40-line reference implementation.

A judge is admitted only if its measured literal-act mean clears `GATE`. That is enforced in code,
not by the caller, so a new judge cannot skip calibration.

### 3. Swap the policy, keep the environment

A policy is any function that calls `env.observe()` and `env.step(judge, operator, n=...)`.
`greedy_agent`, `random_agent` and `ucb_agent` in `environment/env.py` are three worked examples in
under forty lines each.

**Read Finding 2 before writing one.** The environment's null model conditions on *realised*
per-operator sample sizes, so a policy that concentrates budget on the current leader inflates its
own significance bar faster than the extra samples lower it. Greedy clears its null in 0 of 9
cells for this reason; balanced allocation clears 6 of 9. If your policy is doing worse than
random, check its allocation skew before concluding anything about the effect.

## Open questions this environment is already set up to answer

These are ordered by how much they would add and how ready the code is for them.

1. **Is the generational finding real?** Finding 6 shows euphemism ranks about second of six in
   current models of both lineages, against first in every cell of the older frozen panel. Twelve
   live models is too few to settle that. More models per generation, and a larger act bank so
   per-operator n rises, would turn a suggestive gap into a measurable one. Everything needed is
   in `run_live_demo.py --policy balanced`.
2. **Is there a lineage effect underneath the generational one?** Live Western sits at mean rank
   1.83 and live Chinese at 2.33. At six models a side that half-rank gap is not separable from
   noise. This is a sample-size problem, not a design problem.
3. **Does the effect survive a human-validated core set?** Every rewrite here was machine
   generated and machine checked for semantic preservation by two measures that agree at r = 0.72.
   Human adjudication of the disagreeing 7 to 12 percent is the blocker on the EvalScope
   contribution, and it is a self-contained piece of work.
4. **Does a policy with an explicit floor under non-leading operators beat both greedy and
   random?** `--policy balanced` is the crudest possible version. A policy that allocates by
   expected information gain rather than round-robin is an obvious next step and would slot in at
   seam 3.
5. **Does the binary instrument agree with the 0-10 scale?** A known wiring bug leaves the
   judge-provided binary field null for every replay probe; `--live-binary` asks the question
   directly against a live judge. Documented in the README and `run_config.json`.

## What not to do

- **Do not use a Qwen model, or any Alibaba model, as a judge.** A Qwen model generated every
  rewrite in `data/kernel_payload.json`. Judging them with Qwen measures preference leakage, not
  framing sensitivity. This is why no Qwen model appears in any panel here.
- **Do not compare operators from a greedy run.** See seam 3.
- **Do not raise `GATE` to admit a model that fails it.** The gate exists because two IBM Granite
  models rate the unmodified harmful acts as acceptable while their own binary instrument calls
  those same acts wrong. A null result from such a model measures the instrument, not the model.
  `sensitivity_analysis.py` shows judge membership is constant for any gate in [2.642, 5.025).

## Reproducing the published numbers before you change anything

```bash
bash environment/scripts/smoke_test.sh       # no install, no key, no network
bash environment/scripts/reproduce_core.sh   # regenerates campaign_summary.json byte-identically
python3 environment/analyze_live_panel.py --reports "environment/live_panel/report_*.json"
python3 environment/analyze_live_panel.py --reports "environment/live_panel_western/report_*.json"
pytest environment/tests/test_env.py         # 101 tests, all offline
```

If `reproduce_core.sh` does not produce a file identical to the committed
`environment/campaign_summary.json`, something in your environment differs from the one the
results were produced on, and that is worth resolving before building on them.
