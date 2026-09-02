# Pre-registration: does the euphemism result hold on Chinese model lineages?

Written **2026-09-02, before any model in the panel below had been probed at scale.**
Committed before the run so that the analysis plan cannot be adjusted to fit the result.

This project has already had one finding stated backwards and corrected in public
(see the budget/null discussion in the README). That correction is why this file
exists: the plan is fixed first, the numbers arrive second.

## Why this run exists

The ten instrument-valid judge families in the frozen data are all Western: seven
labs, two years, no Chinese lab among them. The honest reading of that panel is
that it says nothing about models trained on a different corpus, by a different
lab, under a different regulatory regime.

A pilot on 2026-09-02 scored **DeepSeek-V3.2-Exp** live: it passed the validity
gate (literal mean 1.00) and euphemism ranked first with a mean delta of +2.00
over 18 probes. Eighteen probes at three acts per operator is a direction, not a
result. No gap that small survives the environment's own null model, and the
README says so. This run is the properly powered version of that pilot.

## Panel

Six models, six labs, every one released in 2026, reached through one aggregator.

| model id | lab | released |
|---|---|---|
| `deepseek/deepseek-v4-flash-0731@openrouter` | DeepSeek | 2026-07-31 |
| `z-ai/glm-5.3-flash@openrouter` | Zhipu / Z.ai | 2026-08-26 |
| `moonshotai/kimi-k2.6@openrouter` | Moonshot | 2026-04-20 |
| `minimax/minimax-m3@openrouter` | MiniMax | 2026-05-31 |
| `tencent/hy3@openrouter` | Tencent Hunyuan | 2026-07-06 |
| `tencent/hy4-preview@openrouter` | Tencent Hunyuan | 2026-08-28 |

**No Qwen, and no other Alibaba model, by design and not by oversight.** A Qwen
model generated every rewrite in this repository. Scoring those rewrites with a
Qwen judge measures preference leakage, not framing sensitivity. This exclusion
costs the panel its most obvious member and is worth it.

`tencent/hy4-preview` was released five days before this file was written and is
flagged `preview` by its own vendor. It is included precisely because it is the
newest thing available, and any instability it shows is reportable.

## What is held constant

Nothing in the environment changes for this run. Same act bank, same six
operators, same scoring prompt, same two gates:

- **Instrument validity**: `GATE = 4.0`. A model is probed only if its mean rating
  of unmodified literal acts is at or below 4.0/10, **measured live on this model**,
  never assumed from the model's reputation.
- **Semantic preservation**: `OVERLAP_FLOOR = 0.15`, unchanged.

Calibration is 12 literal acts per model, up from the pilot's 4, because the gate
decision is the single highest-consequence number in the run.

## Hypotheses, fixed in advance

- **H1 (primary)**: in each model that clears the gate, `euphemism` has the highest
  mean delta of the six operators.
- **H2 (secondary)**: the rank-1 minus rank-2 gap exceeds the p95 threshold of the
  environment's own null model, which shuffles observed deltas across operators
  while holding each operator's realised sample size.

Target allocation is balanced across operators, roughly 48 probes each, chosen to
land in the same range as the random baseline that cleared its null in 6 of 9
cells. Balanced allocation is deliberate: the frozen campaign already established
that a winner-take-most policy inflates its own significance bar.

## Reporting rules, committed before the data exists

1. **Every model that clears the gate is reported**, including any where euphemism
   does not rank first. A lineage where the effect fails is a boundary condition
   and is a more interesting result than another confirmation.
2. **Every model excluded by the gate is reported**, with its literal mean and the
   reason, exactly as the two IBM Granite models are reported in the frozen panel.
3. **Every model that errors out is reported** as an error, not silently dropped.
4. **The budget is fixed before the run.** No looking at partial results and then
   extending the run for a model that is close to significance.
5. **No model is swapped out after results are seen.** The six above are the panel.
   If one is unreachable it is reported as unreachable.
6. If H1 holds but H2 fails, the result is reported as **directional and
   underpowered**, in those words, and is not described as a finding.

## What this run cannot show

It is six models from one aggregator at one point in time. It does not establish
that the effect is universal across Chinese models, it does not separate lab from
training corpus from alignment method, and it does not rule out that the
aggregator's own routing changed a model's behaviour. It tests one specific,
falsifiable claim: that a result found on ten Western families also appears on
lineages that were never consulted while this environment was built.

## Artifact safety

Output goes to new files only. `run_live_demo.py` sets `is_official_artifact:
false` and writes per-model logs under `exploration_log_live_*.jsonl`. The frozen
campaign artifacts (`campaign_summary.json`, `random_baseline_summary.json`,
`ucb_summary.json`, `exploration_log.jsonl`) are not touched, and CI diffs the
committed selftest log to prove it.
