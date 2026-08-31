# The exploration environment

**The question this environment explores:** does a language model's moral verdict on a
harmful act change when only the wording changes and the act itself stays the same? Take an
act a judge model rates clearly wrong, rewrite it under one of six framing operations from
critical-discourse-analysis (euphemism, agent-deletion, nominalization, and others), and
re-score the rewrite blind. Which operator moves the verdict, against which judge, and by how
much, is what an agent exploring this environment gets to probe. The repository root above
this directory holds the finished, static version of that experiment: fixed stimuli, fixed
judges, a result table. This directory turns the same underlying material (the act bank, the
six framing operators, the replay-judged data) into something an agent can *act in*, where its
choice of action changes what it observes next, rather than something you just read a table
from.

## What's fixed, what's explorable, what's fed back

```
FIXED       the act bank, the judge panel, the validity gate (literal mean <= 4.0/10),
            the semantic-preservation floor (content-word overlap >= 0.15), lineage
            separation (rewrite-generator != judge, so a judge never marks its own homework)
EXPLORABLE  which operator, against which judge, at what sample size, in which order
FEEDBACK    per-act delta, binary flip, the rank-1-to-rank-2 gap with confidence interval,
            the validator's verdict and reason, budget remaining
```

## Run it

No API key, no network, no cost. Everything runs in **replay mode** against the frozen
judged data already in this repo: every score returned is a real recorded judgment from a
real model, just not a fresh one.

```bash
python3 env.py --selftest              # one seed, budget=300, ~84 steps, prints a trace
python3 run_campaign.py                # 3 seeds, budget=2000 each, writes campaign_summary.json
```

Both are pure Python standard library. `env.py` finds `data/recency_results.json` and
`data/kernel_payload.json` relative to the repo root automatically.

## The agent in here is deliberately simple, and its bug is on the record

`greedy_agent()` probes every unprobed (judge, operator) cell once (breadth), then spends
the rest of its budget deepening whichever operator currently leads for the judge it's
looking at, rotating round-robin across every instrument-valid judge so each one gets
deepened in turn.

That round-robin wasn't the first version. The original policy deepened
`judges_valid[0]`, the alphabetically-first judge, for the entire depth phase, and never
came back to the others. It's a real exploitation bug: those other judges' `gap_to_second`
stayed stuck at breadth-phase sample sizes for the whole run, once you can already tell
they've been under-explored by looking at the coverage. `campaign_summary.json` is the
rerun after the fix: `final_gap_by_judge` now reports every valid judge with comparable
sample sizes (roughly 400-500 probes each across ~1400 depth-phase steps per seed, vs. all
of it going to one judge before), not just one.

The bug and the fix are left visible on purpose. The exploration process being inspectable
enough to catch this kind of thing is closer to what this environment is *for* than a
smoothed-over agent would be.

## What the campaign found across seeds

Read directly from `campaign_summary.json`'s `final_gap_by_judge` and `null_model_by_judge`:
**none of the 9 (judge, seed) gaps clear their own seed's null-model p95 threshold**, not
even `gemma4-12b`'s. What differs is how close each gets. `gemma4-12b` picks euphemism at
rank-1 in 2 of 3 seeds, reaching 29-71% of its own threshold each time (0.617/2.1, 1.971/3.064,
1.765/2.5). `gemma4-e4b` and `olmo3-7b` disagree with each other and across seeds
(nominalization, euphemism, functionalization all appear at rank-1 at least once for one of
them), sitting at only 2-18% of their thresholds. Stated precisely: this campaign's budget
does not have the statistical power to confirm any single-judge preference against chance,
but the relative signal strength differs enough between `gemma4-12b` and the other two that
it's a candidate signal worth a higher-budget follow-up, not a confirmed finding. That the two
smaller-effect judges disagree with each other and with `gemma4-12b` is itself worth noting,
not a "euphemism wins everywhere" claim the data doesn't support.

This is consistent with, but not identical to, the headline `+3.264` euphemism delta in the
top-level README for `Gemma-4-12B`: that number is the full N=81 aggregate-table result;
this environment's per-act paired deltas over a budget-limited campaign are a different,
smaller-sample statistic computed a different way. Reconciling the two exactly is open work,
noted here rather than left implicit.

## A random-exploration baseline, for comparison against the policy above

`run_random_baseline.py` runs `random_agent()` (also in `env.py`): a uniformly random
(judge, operator) pick every step, no breadth/depth logic. Same 3 seeds, same budget=2000 as
the real campaign, writing to its own files so nothing above is touched or overwritten.

Under random exploration, euphemism is rank-1 in **8-9 of 9** (judge, seed) cells: it's
simply the strongest single operator on this bank, findable by undirected breadth alone. The
real greedy policy above only picks euphemism in **5 of 9** cells. Read that correctly: the
policy's actual exploration value is the cells where it *diverges* from that obvious default
and finds a genuinely different, judge-specific leader instead — which is exactly what the
round-robin fix above made visible in the first place, not a weaker result than random search.
One counter-example, stated plainly rather than hidden: seed 42/`olmo3-7b`, random's gap was
smaller than greedy's. No null-model significance check has been run against the random
numbers yet.

## Run it, one-click entry points

```bash
bash scripts/smoke_test.sh        # wraps env.py --selftest, checks required data files first
bash scripts/reproduce_core.sh    # wraps run_campaign.py, prints each seed's final gaps at the end
```

Both are thin wrappers around the commands above — no new logic, just sanity checks and a
readable pass/fail instead of a raw traceback if something's missing.

## Files

```
env.py                              the environment: Bank, ReplayJudge, Environment, greedy_agent, random_agent, selftest
run_campaign.py                     the non-toy run: budget=2000, seeds 0/1/42, writes campaign_summary.json
run_random_baseline.py              the random-exploration baseline described above
campaign_summary.json               per-seed summary: cells probed, gaps, rejection rate, null-model baseline
random_baseline_summary.json        same, for the random-exploration baseline
exploration_log.jsonl               the 84-record selftest trace (seed=0, budget=300)
exploration_log_campaign_seed*.jsonl   the full immutable per-step log for each campaign seed
exploration_log_random_seed*.jsonl     the full immutable per-step log for each random-baseline seed
scripts/smoke_test.sh, reproduce_core.sh   one-click entry points wrapping the commands above
```

Every step, in either mode, is appended to its log file as one immutable JSON record.
Rejected probes (content drift below the overlap floor) are logged with their rejection
reason rather than dropped, because the rejection rate is itself evidence about how much
room a rewrite operator has to drift before it stops being the same act.
