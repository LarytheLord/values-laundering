# Contributing

This started as a single-author study and is now an exploration environment
other people can run and extend. Both kinds of contribution are welcome, and
they have different bars.

## Before anything else

```bash
bash environment/scripts/smoke_test.sh
```

No install, no API key, no network. If that fails on a fresh clone, that is a
bug worth reporting on its own — the environment replays frozen judged data
and is supposed to run on a bare `python3`.

## The one rule that matters here

**Numbers in the documentation must be recomputed from the committed data, not
carried over from a previous draft.** This repository has had real bugs where a
README claim and the underlying JSON disagreed, and they were only caught by
someone re-deriving the number by hand. If you change anything that touches a
reported figure, regenerate it and say in the commit message which file you
regenerated it from.

The same applies to claims about what the environment measures. `threshold_flip`
exists because `binary_verdict_pair` was documented as a working signal while
being silently `null` for every probe. If you find something like that, the
expected response is to document it precisely, not to quietly patch the number.

## Extending the environment

The most useful directions, roughly in order:

- **A new rewrite operator.** The six in `MOVES` come from van Leeuwen's social
  actor representation taxonomy. Adding a seventh means adding rewrites for it
  across the act bank, not just a key in the dict.
- **A different exploration policy.** `greedy_agent` and `random_agent` share
  one `observe()`/`step()` loop; a new policy is a third function with the same
  shape. `random_agent` exists as the reference frame — a new policy should be
  compared against it, not against nothing.
- **A live-inference mode.** The environment is replay-only today, which is why
  it is free and deterministic and also why it cannot see a model's behaviour
  change over time. A live mode behind a flag, leaving replay as the default,
  would close a real limitation.

## Tests

```bash
pip install pytest
pytest environment/tests/test_env.py -v
```

Each test class in that file exists because this codebase produced the
corresponding bug at some point — a gate boundary, a policy that stopped
rotating between judges, a field that returned a plausible-looking default
instead of `None`. New tests are most valuable when they are shaped after a
specific failure rather than after a function signature.

## Data

`data/` holds the frozen judged outputs the environment replays. Regenerating
it requires GPU inference (`kernels/`) and is not part of the normal
contribution path. Treat those files as fixtures: if a change requires editing
them, that is worth discussing in an issue first, because every committed
result in the repository is computed from them.

## License

Contributions are accepted under the MIT license in `LICENSE`.
