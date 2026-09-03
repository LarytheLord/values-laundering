# Western control: the effect tracks model generation, not lineage

Run 2026-09-03 against the post-hoc control described in
`../CHINESE_PANEL_PREREGISTRATION.md` (Amendment). That amendment was written **after** the
Chinese results were known and says so, and it fixed in advance what each outcome would mean.
This is the outcome it called "the one that costs us the current headline".

## Why this run exists

The Chinese panel returned euphemism rank-1 in 2 of 6, against 9 of 9 in the frozen Western
panel. Read as written, that says the effect is lineage-dependent. But the two panels differ on
**three axes at once**: lineage, model recency, and replay versus live. A reviewer is entitled to
say the gap has nothing to do with lineage.

So: six current Western models, six labs, the identical live loop, balanced allocation, the same
bank, the same prompt, the same two gates. Tier matched deliberately, since the Chinese panel was
mostly flash and small tier.

## Result

| model | lab | literal mean | gate | rank-1 | euphemism rank | euphemism delta |
|---|---|---|---|---|---|---|
| mistral-medium-3-5 | Mistral | 1.42 | pass | **euphemism** | 1 | +2.733 |
| mistral-small-2603 | Mistral | 2.75 | pass | **euphemism** | 1 | +2.533 |
| gpt-5.6-luna-pro | OpenAI | 1.00 | pass | **euphemism** | 1 | +1.433 |
| grok-4.6 | xAI | n/a | pass | **euphemism** | 1 | +0.867 |
| gemini-3.7-flash | Google | 0.44 | pass | necessity | 3 | +0.438 |
| claude-sonnet-5 | Anthropic | 1.67 | pass | functionalization | 4 | +0.423 |

All six cleared the validity gate. Euphemism ranked first in 4 of 6, and 4 of 6 gaps cleared
their own null.

## The comparison that matters

Rank-1 counts throw away information: they record only which operator won, not where euphemism
placed when it lost. Mean euphemism rank across all six operators uses the whole ordering, and
chance is 3.5.

| panel | mode | models | euphemism rank-1 | **mean euphemism rank** | mean euphemism delta |
|---|---|---|---|---|---|
| frozen Western, 10 families | replay | 9 cells | **9 of 9** | **1.00** | n/a |
| live Chinese | live | 6 | 2 of 6 | **2.33** | +1.317 |
| live Western | live | 6 | 4 of 6 | **1.83** | +1.404 |

**The two live panels are close to each other, and both sit far from the frozen panel.**

## What this means, corrected

The lineage reading is **wrong**, and this run is what showed it. Euphemism does not rank first
in most current models of *either* lineage. It sits around rank 2 of 6 in both, which is well
above chance but is not the dominance the frozen panel shows.

The axis that actually separates the results is **model generation**, not nationality:

- Older, mostly open-weight models, scored in replay: euphemism ranks first in every cell.
- Current frontier-tier models of either lineage, scored live: euphemism ranks around second,
  and which operator wins varies by model.

So the honest statement is that **framing sensitivity to euphemism specifically has attenuated in
current models, and the frozen panel's 9-of-9 is a fact about that generation of models rather
than about the West.** The broader effect has not vanished: euphemism still beats chance in both
live panels, and every live model still moves its verdict under some rewrite operator.

## What this does not show

It does not show current models are robust to framing. Every model in both live panels has some
operator with a positive mean delta, and in several cases a large one. What changed is *which*
operator leads, not whether wording moves the verdict.

Nor is it a clean generational law. Twelve live models across two lineages is a small sample, the
bank is single-domain, and mean rank at n=27 to 47 per operator is a coarse instrument. What it
is enough for is to reject the lineage explanation we would otherwise have published.

## A note on availability in mainland China

The Chinese panel is the one a mainland reviewer can reproduce natively: DeepSeek, Zhipu GLM,
Moonshot Kimi, MiniMax and Tencent Hunyuan are all served domestically. Of the Western control,
the Mistral models are open-weight and obtainable, while Claude, GPT and Gemini are not reachable
from the mainland. That is precisely why this set is presented as a **control for a confound**
rather than as a headline panel: its job was to test whether recency explained the Chinese
result, and it did.

## Reproducing

```bash
python3 environment/analyze_live_panel.py \
  --reports "environment/live_panel_western/report_*.json" \
  --out environment/live_panel_western/western_control_analysis.json
```

Offline, no key. Regenerating the reports needs an API key and `--policy balanced`.
