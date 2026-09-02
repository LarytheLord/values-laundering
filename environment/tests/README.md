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
second. That includes the live-mode tests: every one of them injects a fake
client, and `TestLiveModeNeverTouchesTheNetwork` additionally monkeypatches
`urllib.request.urlopen` to raise, so a future refactor that bypassed the
injected client would fail loudly here rather than silently dialling out from
a CI runner.

## What each test class defends against

- **TestInstrumentValidityGate** -- a judge whose literal-act mean fails the
  4.0/10 gate leaking into `judges_valid` (or a passing judge being wrongly
  excluded), including the exact boundary at 4.0 and the case where a
  missing `base` score should be dropped from the mean, not treated as 0.
- **TestOverlapFloor** -- the semantic-preservation floor (`OVERLAP_FLOOR`)
  accepting a rewrite that drifted too far from its source act, or
  rejecting one that didn't, including the exact boundary ratio.
- **TestGreedyAgentDepthPhaseRoundRobin** -- regression test for a real
  exploitation bug this codebase had, where the depth phase always deepened
  `judges_valid[0]` and never rotated to the other valid judges. Asserts an
  even 4/4/4 split across three judges instead of one judge absorbing every
  depth round.
- **TestBinaryVerdictPairMissingBinField** -- regression test for a real dead
  field this codebase had: no per-operator `moves` entry in the real data
  carries a `bin` key, so the second element
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

### UCB policy

- **TestUCBAgentUsesEnvironmentRNG** -- same guarantee as
  `TestRandomAgentUsesEnvironmentRNG`, for `ucb_agent`'s tie-breaking: a
  policy reaching for the global `random` module would void the
  same-seed-same-trajectory property for that policy alone, and so
  invalidate every cross-policy comparison it appears in.
- **TestUCBAgentInitialization** -- UCB1's bound only holds if every arm is
  pulled once before any arm is pulled twice; a policy that skipped the
  initialization sweep would look like a bandit while behaving like
  something else.
- **TestUCBRewardNormalization** -- `normalize_delta` must map `[-10, +10]`
  onto `[0, 1]` affinely and order-preservingly, because the environment
  ranks operators by raw mean delta and a policy optimizing a differently
  ordered reward could converge on a different rank-1 than the one the
  environment reports.
- **TestUCBAgentHandlesCellsThatNeverScore** -- a cell where every act is
  rejected by the overlap floor still counts as a pull, or the
  initialization phase loops on it forever.

### Live mode

All of these are offline. They defend the claim that `LiveJudge` is a genuine
drop-in and that live mode cannot leak a credential.

- **TestLiveJudgeIsDropInForReplayJudge** -- the whole design claim is that
  `Environment` needs no change to accept a live judge. Asserted
  structurally (public surface and `score()` dict keys compared against
  `ReplayJudge`) rather than by one happy-path example, then exercised for
  real: `Environment` accepts one, the shipped `greedy_agent` runs against
  one, and the resulting log carries the identical schema a replay run
  produces. A claim like this rots silently unless a test pins it.
- **TestLiveScoreParsing** -- the 0-10 parser is the highest-risk piece of
  live mode. Falcon-H1R once returned 81/81 unparseable scores because a
  reasoning preamble ate the token budget, and the frozen data lost a whole
  judge family to it. Covers the bare number, both scale boundaries, `10`
  not being read as `1`, a rambling answer (verdict is last, not first), a
  `<think>` preamble, and -- most importantly -- that an unparseable reply
  returns `None` rather than defaulting to `0`, which on this scale means
  "completely unacceptable" and is a real, strong judgment.
- **TestLiveJudgeCalibrationAndGate** -- instrument validity must be
  *measured* live on the same 4.0 gate, never assumed because the caller
  named a model. A model that rates the literal cruelty acts as acceptable
  is excluded exactly as in replay, and `Environment.step` refuses it.
- **TestLiveJudgeScoring** -- delta arithmetic, `threshold_flip` failing
  closed to `None` on a missing score, and that the *rewrite* text is what
  gets sent. That last one matters: `LiveJudge` keeps `ReplayJudge`'s
  two-argument signature and resolves the rewrite text from the bank
  itself, so a broken lookup would score the literal act twice and make
  every delta `0` -- a failure that reads like a finding rather than a bug.
  It also pins the exact scoring-prompt string, so editing it becomes a
  deliberate act that breaks a test.
- **TestLiveJudgeCachesLiteralScores** -- one literal score per act, not one
  per operator. Getting this wrong would sextuple the cost of every live
  campaign while changing no number, which nothing else in the suite would
  notice.
- **TestLiveJudgeBinaryInstrument** -- live mode answers the question replay
  cannot. `binary_verdict_pair[1]` is null for every replay probe because of
  the `euph_bin` wiring bug; with `--live-binary` the binary question is
  asked directly, with the frozen `BIN` prompt, for all six operators.
- **TestLiveModeNeverLeaksTheApiKey** -- this project has had three separate
  credential-leak incidents, so the guarantee gets a test rather than a
  code-review habit. The key must be reachable only from the process
  environment, and must not survive into a `repr`, an exploration-log
  record, or an exception message; the missing-key error names the
  variables it tried and points at the offline escape hatch, never a value.
- **TestLiveModeNeverTouchesTheNetwork** -- CI has no key and no outbound
  network and must never need either.
- **TestLiveBaselineReportsItsOwnMode** -- `baseline_no_intervention` used to
  hardcode `mode="replay"`. With a live judge attached that string was
  simply false, and a reader trusting it would have mis-attributed a live
  run's zero noise floor to replay.
