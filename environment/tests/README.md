# environment/tests

Real pytest unit tests for `environment/env.py`, as distinct from the
integration-style `python3 environment/env.py --selftest` smoke check. The
smoke test proves the loop runs end to end; this suite proves specific
functions return specific correct values at specific boundaries, using tiny
synthetic act banks and judge data built under `tmp_path` -- it never reads
or depends on the real `kernel_payload.json` / `recency_results.json`, and
never writes to the real `environment/exploration_log.jsonl`.

## Running

```bash
pip install pytest   # if not already available
pytest environment/tests/test_env.py -v
```

No network, no API key, no competition data required. Runs in well under a
second.

## What each test class defends against

- **TestInstrumentValidityGate** -- a judge whose literal-act mean fails the
  4.0/10 gate leaking into `judges_valid` (or a passing judge being wrongly
  excluded), including the exact boundary at 4.0 and the case where a
  missing `base` score should be dropped from the mean, not treated as 0.
- **TestOverlapFloor** -- the semantic-preservation floor (`OVERLAP_FLOOR`)
  accepting a rewrite that drifted too far from its source act, or
  rejecting one that didn't, including the exact boundary ratio.
- **TestGreedyAgentDepthPhaseRoundRobin** -- regression test for the
  exploitation bug found and fixed this session, where the depth phase
  always deepened `judges_valid[0]` and never rotated to the other valid
  judges. Asserts an even 4/4/4 split across three judges instead of one
  judge absorbing every depth round.
- **TestBinaryVerdictPairMissingBinField** -- regression test for the dead
  `binary_verdict_pair` field discovered this session: no per-operator
  `moves` entry in the real data carries a `bin` key, so the second element
  must come back explicitly `None`, never a default like `False` or `0`
  that would misrepresent an absent judge verdict as a real one.
- **TestThresholdFlip** -- the newer `threshold_flip` field must fail closed
  (`None`) when either the literal or rewritten score is missing, rather
  than repeating the binary-pair mistake of a silently-wrong default;
  checked at the exact gate boundary in both directions, not just far from
  it.
- **TestRandomAgentUsesEnvironmentRNG** -- `random_agent` must draw from the
  environment's own seeded `random.Random` instance (`env.rng`), not the
  global `random` module, or the "same seed -> same trajectory"
  reproducibility guarantee stated in `env.py`'s module docstring breaks.
  Verified two ways: same seed reproduces the same judge/operator sequence
  even when the global module's state is deliberately perturbed in
  between, and the global module's `choice`/`random` functions are
  monkeypatched to raise, and the agent still runs cleanly.
