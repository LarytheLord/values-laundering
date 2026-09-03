# Same-generation live panel: ruling out replay as the explanation

Run 2026-09-03. This is the second of two controls on Finding 6, and it closes the confound the
first one left standing.

## Why this run exists

The Western control showed the Chinese/Western gap was not a lineage effect: both live panels sat
around euphemism rank 2, far from the frozen panel's rank 1. That pointed at model generation.

But it did not establish that. The frozen panel and both live panels still differed on **two**
things at once: model generation *and* scoring mode (replay of stored judgments versus live API).
"Older models" was still entangled with "replay".

The clean way to separate them is to run **frozen-era models live**. If they behave like the
frozen panel, mode is not the explanation and generation is. If they behave like the current
models, the whole difference was an artifact of replay.

## Panel

Four models from the same generation as the frozen judges, three of which appear in the frozen
ten-family panel itself. Identical live loop, balanced allocation, both gates unchanged.

| model | relation to frozen panel | live literal mean | gate | rank-1 | euphemism rank |
|---|---|---|---|---|---|
| gpt-oss-20b | in the frozen panel | 0.75 | pass | **euphemism** | 1 |
| gemma-4-31b-it | same family as two frozen judges | 1.33 | pass | **euphemism** | 1 |
| granite-4.1-8b | **is** a frozen judge (gate-excluded there) | 1.83 | pass | **euphemism** | 1 |
| gpt-oss-safeguard-20b | in the frozen panel | 1.08 | pass | functionalization | 2 |

Euphemism ranked first in **3 of 4**, mean euphemism rank **1.25**.

## The four-cell result

| condition | mode | per-operator n | mean euphemism rank |
|---|---|---|---|
| frozen-era models, full bank | replay | 68 to 81 | **1.00** |
| frozen-era models, subsampled | replay | 30 | **1.05** |
| **frozen-era models** | **live** | 27 to 47 | **1.25** |
| current-generation Western | live | 27 to 47 | **1.83** |
| current-generation Chinese | live | 27 to 47 | **2.33** |

Two alternative explanations are now tested and rejected:

- **Sample size.** Subsampling the frozen judges to n=30 leaves euphemism at mean rank 1.05, not
  1.83 or 2.33. Small n does not produce the live panels' weaker ranking.
  (`../subsample_frozen_panel.py`, offline, no key.)
- **Replay versus live.** Frozen-era models scored live land at 1.25, close to their own replay
  value of 1.00 and far from current models at 1.83 and 2.33. Mode does not produce it either.

What remains is **model generation**, and it is now the only surviving explanation rather than
the first one that occurred to me.

## 🔴 An uncomfortable observation that must be reported

`granite-4.1-8b` is one of the two models the frozen data **excludes** by the instrument-validity
gate: its frozen literal-act mean is 5.02, above the 4.0 gate. Run live here, the same model name
scores **1.83** and passes comfortably.

That is a factor-of-nearly-three difference in the number the gate decides on, for a model that is
used in this submission as a worked example of the gate doing its job.

I am not going to explain it away. Candidate causes, none of which I have tested:

- version drift behind the same model id on a hosted endpoint,
- a different serving path, quantisation, or default sampling than the frozen collection used,
- something specific to the frozen Granite collection itself.

**What this does and does not change.** It does not weaken the gate's justification, because the
justification never rested on the literal mean alone: the frozen Granite models rate the
unmodified harmful acts as acceptable *while their own binary instrument calls those same acts
wrong on 76 of 77 and 77 of 81*. That internal contradiction is present in the frozen data
regardless of what a live re-measurement says. What it does change is the scope of the claim: the
exclusion is a property of **that frozen measurement**, not a permanent property of the model, and
the report and slides now say so.

It is also, read another way, a small finding in its own right: instrument validity is not a fixed
attribute of a model id. It has to be measured on the same serving path as the results it gates,
which is exactly why this environment measures it live rather than assuming it.

## Limits

Four models, and only two of them (`gpt-oss-20b`, `gpt-oss-safeguard-20b`) are strictly the same
model as a frozen panel member. `gemma-4-31b-it` is a sibling of the frozen Gemma judges, not one
of them, and the frozen campaign's own three judges (gemma-4-12B-it, gemma-4-E4B-it,
Olmo-3-7B-Instruct) are not available on the aggregator used here. A run against those exact three
would be the stronger version of this control and is the first thing I would do next.

Two further models, `gpt-oss-120b` and a second `gpt-oss-20b` pass, were launched and stalled at
the provider before completing. They are not reported, and no partial result from them is used.

## Reproducing

```bash
python3 environment/analyze_live_panel.py \
  --reports "environment/live_panel_samegen/report_*.json" \
  --out environment/live_panel_samegen/samegen_analysis.json
python3 environment/subsample_frozen_panel.py
```

Both offline, no API key.
