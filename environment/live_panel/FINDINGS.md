# Chinese-lineage panel: the euphemism result does not generalise

Run 2026-09-02 against the plan fixed in `../CHINESE_PANEL_PREREGISTRATION.md`,
which was committed (`ef2b8c4`) before a single API call was made.

## Result

Six models, six labs, every one released in 2026, all reached live through one
aggregator. Balanced allocation, so every operator ends with the same realised n.

| model | lab | literal mean | gate | rank-1 | gap | null p95 | clears null |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4-flash-0731 | DeepSeek | 0.42 | pass | **euphemism** | 0.749 | 0.690 | **yes** |
| tencent/hy3 | Tencent | 0.42 | pass | functionalization | 0.659 | 0.618 | **yes** |
| moonshotai/kimi-k2.6 | Moonshot | 0.42 | pass | necessity | 0.150 | 0.922 | no |
| minimax/minimax-m3 | MiniMax | 0.83 | pass | **euphemism** | 0.100 | 0.582 | no |
| tencent/hy4-preview | Tencent | 1.08 | pass | necessity | 0.100 | 1.460 | no |
| z-ai/glm-5.3-flash | Zhipu | 2.00 | pass | functionalization | 0.125 | 1.186 | no |

**6 of 6 cleared the instrument-validity gate. Euphemism ranked first in 2 of 6.
2 of 6 gaps cleared their own null. The rank-1 operator splits three ways, two
models each.**

The comparison that matters is against the frozen Western panel run under the
same balanced allocation, where **euphemism ranked first in 9 of 9 judge-seed
cells** and 6 of 9 gaps cleared their null.

## What this means, stated no more strongly than the data allows

**The euphemism result is lineage-dependent.** It was established on ten Western
families across seven labs, and it does not carry over to these six Chinese
models as a general claim.

It is not simply absent either. On `deepseek-v4-flash`, the most prominent
Chinese lineage here, euphemism ranks first with a gap that clears its own null.
An earlier 18-probe pilot on a different DeepSeek model, `DeepSeek-V3.2-Exp`,
also put euphemism first. Two DeepSeek models, the same answer. On the other
five the leading operator is something else, and on four of them no gap clears
noise at all.

**This is not an artifact of sample size.** Two reasons. First, which operator
ranks first does not depend on n; n governs whether a gap is significant, not
who leads. Second, the null thresholds here (0.58 to 1.46) sit in the same range
as the frozen balanced baseline's (0.39 to 0.52), so the two runs are broadly
comparable in power. The higher thresholds on `hy4-preview` and `glm-5.3-flash`
reflect genuinely more variable deltas from those two models, not less data.

**What cannot be concluded.** That these models are more robust to framing. A
diffuse effect and a genuinely smaller effect look the same at this n, and six
models is a small panel. Distinguishing them needs a larger bank per operator
and more models per lab.

**One gate note worth reporting.** `glm-5.3-flash` passes at a literal mean of
2.00, comfortably inside the 4.0 gate but the highest in this panel and roughly
five times the lowest. It rates the unmodified harmful acts as less clearly
wrong than the others do while still passing. That is inside the instrument's
tolerance, and it is the kind of thing the gate exists to make visible.

## Why the run happened at all

The frozen panel is entirely Western: seven labs, two years, no Chinese lab.
The honest reading was always that it said nothing about models trained on a
different corpus by a different lab. Rather than continue to say so as a
limitation, the environment was pointed at six such models with no change of any
kind to the bank, the operators, the prompt, or either gate. Only the judge
changed.

The environment returned an answer its author did not want, in one evening, for
under a dollar, on lineages it was never built against. That is the point of
building an instrument rather than a demonstration.

## Reproducing

```bash
python3 environment/run_live_demo.py \
  --models "deepseek/deepseek-v4-flash-0731@openrouter" \
  --policy balanced --calibration-n 12 --budget 288 --rounds 288 --n 1
python3 environment/analyze_live_panel.py --reports "environment/live_panel/report_*.json"
```

The analysis step is fully offline and needs no key: it reads the committed
reports. Only regenerating them needs an API key, set in `OPENROUTER_API_KEY`.

`--policy balanced` matters. The default is greedy, which reproduces the frozen
campaign and is the right default, but it concentrates budget on the current
leader. An earlier attempt at this panel on greedy allocation produced n=194 on
one operator and n=1 on four others, and no cross-operator rank claim can be
built on that.
