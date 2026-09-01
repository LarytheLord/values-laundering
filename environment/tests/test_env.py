"""Real (non-smoke) unit tests for environment/env.py.

Why this file exists at all: env.py had only an integration-style
`--selftest` smoke check before this file, which proves the loop runs but
would not catch a wrong constant, an inverted comparison, or a field that is
silently always None. Every test class below is aimed at one of the bug
classes this exact codebase actually produced, or at a guarantee it now
depends on:
  (1) the validity gate's boundary/exclusion logic,
  (2) greedy_agent's depth phase locking onto one judge instead of round-robin,
  (3) a judge-provided field (binary_verdict_pair) that is silently always
      null because of a key mismatch, discovered late by manual inspection,
  (4) the newer threshold_flip field, which needs to fail closed (None) on
      missing data rather than defaulting to a value that looks meaningful.
  (5) a policy reaching for the global `random` module instead of env.rng,
      which would silently void the same-seed-same-trajectory guarantee for
      that policy alone and so invalidate any cross-policy comparison, and
  (6) ucb_agent's UCB1 initialization sweep and its bounded reward scale,
      either of which failing quietly would leave the policy looking like a
      bandit while behaving like something else.
A generic "does it run" test would have caught none of these; each test here
is shaped after the specific failure, not after the function signature.

env.py is loaded via importlib rather than a normal package import because
this project has no __init__.py / package layout -- it is a single script.
Every test builds its own tiny synthetic act bank and judge data via
tmp_path rather than touching the real kernel_payload.json /
recency_results.json, so the suite is fast, has no dependency on the
competition data's actual contents, and (per the read-only constraint on
env.py during concurrent edits elsewhere) never writes to the real
environment/exploration_log.jsonl -- every Environment built here is given
an explicit log_path under tmp_path.
"""
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ENV_PATH = Path(__file__).resolve().parent.parent / "env.py"
_spec = importlib.util.spec_from_file_location("env_under_test", ENV_PATH)
env = importlib.util.module_from_spec(_spec)
sys.modules["env_under_test"] = env
_spec.loader.exec_module(env)

GATE = env.GATE
OVERLAP_FLOOR = env.OVERLAP_FLOOR


# --------------------------------------------------------------- test helpers

def _write_kernel_payload(tmp_path, act_texts):
    """A minimal stand-in for kernel_payload.json: one 'exp1' list, each
    record's rewrite text equal to its own act text so overlap() never
    rejects it -- tests that need to exercise the overlap gate build their
    own act/rewrite pairs directly instead of going through this helper."""
    acts = [{"act": t, "moves": {m: t for m in env.MOVES}} for t in act_texts]
    p = tmp_path / "kernel_payload.json"
    p.write_text(json.dumps({"exp1": acts}))
    return str(p)


def _write_recency_results(tmp_path, families):
    p = tmp_path / "recency_results.json"
    p.write_text(json.dumps(families))
    return str(p)


def _acts_block(act_texts, base, rewritten, delta, base_bin=1, include_bin=False):
    block = {}
    for t in act_texts:
        moves = {}
        for m in env.MOVES:
            entry = {"score": rewritten, "delta": delta}
            if include_bin:
                entry["bin"] = 0
            moves[m] = entry
        block[t] = {"base": base, "base_bin": base_bin, "moves": moves}
    return block


def make_replay_judge(tmp_path, family, acts, model="test-model"):
    _write_recency_results(tmp_path, {family: {"model": model, "acts": acts}})
    return env.ReplayJudge(family, data_dir=str(tmp_path))


@pytest.fixture
def make_single_judge_env(tmp_path):
    """Factory fixture: build a one-judge, one-or-more-act Environment from
    explicit base/rewritten/delta values, so each test controls exactly the
    numbers its assertion depends on instead of inheriting shared fixture
    data that might not hit the case being tested."""

    def _make(judge_key, act_texts, base, rewritten, delta,
              budget=20, include_bin=False, base_bin=1):
        bank_path = _write_kernel_payload(tmp_path, act_texts)
        acts = _acts_block(act_texts, base=base, rewritten=rewritten,
                            delta=delta, base_bin=base_bin, include_bin=include_bin)
        _write_recency_results(tmp_path, {judge_key: {"model": "m", "acts": acts}})
        bank = env.Bank(path=bank_path)
        judge = env.ReplayJudge(judge_key, data_dir=str(tmp_path))
        return env.Environment([judge], bank=bank, budget=budget, seed=0,
                                log_path=str(tmp_path / f"{judge_key}_log.jsonl"))

    return _make


SIX_ACT_TEXTS = [
    f"synthetic policy act number {i} carries distinct filler tokens "
    f"alphaword{i} bravoword{i} charlieword{i} deltaword{i}"
    for i in range(6)
]


@pytest.fixture
def three_valid_judges_bank_and_judges(tmp_path):
    """Three instrument-valid judges (literal mean 2.0, well under the 4.0
    gate) sharing one six-act synthetic bank -- enough cells (3 judges x 6
    operators = 18) for greedy_agent's breadth phase to exhaust in a known
    number of rounds, so the depth phase that follows is isolated and
    countable."""
    bank_path = _write_kernel_payload(tmp_path, SIX_ACT_TEXTS)
    families = {
        jk: {"model": f"{jk}-model",
             "acts": _acts_block(SIX_ACT_TEXTS, base=2.0, rewritten=6.0, delta=1.0)}
        for jk in ("j0", "j1", "j2")
    }
    _write_recency_results(tmp_path, families)
    bank = env.Bank(path=bank_path)
    judges = [env.ReplayJudge(jk, data_dir=str(tmp_path)) for jk in ("j0", "j1", "j2")]
    return bank, judges


# ------------------------------------------------------ instrument validity gate

class TestInstrumentValidityGate:
    """Defends against: a judge whose literal-act mean fails the gate (>4.0)
    leaking into judges_valid, or a passing judge being wrongly excluded."""

    def test_mean_exactly_at_gate_is_valid(self, tmp_path):
        judge = make_replay_judge(
            tmp_path, "at_gate",
            {"act one": {"base": GATE, "base_bin": 0, "moves": {}},
             "act two": {"base": GATE, "base_bin": 0, "moves": {}}},
        )
        assert judge.literal_mean == GATE
        assert judge.instrument_valid is True

    def test_mean_just_above_gate_is_excluded(self, tmp_path):
        judge = make_replay_judge(
            tmp_path, "above_gate",
            {"act one": {"base": GATE + 0.01, "base_bin": 0, "moves": {}},
             "act two": {"base": GATE + 0.01, "base_bin": 0, "moves": {}}},
        )
        assert judge.instrument_valid is False

    def test_mean_below_gate_is_valid(self, tmp_path):
        judge = make_replay_judge(
            tmp_path, "below_gate",
            {"act one": {"base": GATE - 2.0, "base_bin": 0, "moves": {}}},
        )
        assert judge.instrument_valid is True

    def test_acts_with_missing_base_are_excluded_from_the_mean(self, tmp_path):
        # If the None entry were not filtered, statistics.mean would raise or
        # the mean would be skewed; base=None must be dropped, not treated as 0.
        judge = make_replay_judge(
            tmp_path, "partial_base",
            {"scored": {"base": GATE, "base_bin": 0, "moves": {}},
             "unscored": {"base": None, "base_bin": None, "moves": {}}},
        )
        assert judge.literal_mean == GATE
        assert judge.instrument_valid is True

    def test_environment_splits_valid_and_excluded_judges(self, tmp_path, make_single_judge_env):
        # Build one valid and one excluded judge sharing the same tmp_path bank file,
        # then compose an Environment by hand (the factory fixture only wires one judge).
        act_texts = ["shared act for gate split test"]
        bank_path = _write_kernel_payload(tmp_path, act_texts)
        families = {
            "ok": {"model": "m", "acts": _acts_block(act_texts, base=2.0, rewritten=6.0, delta=1.0)},
            "bad": {"model": "m", "acts": _acts_block(act_texts, base=9.0, rewritten=9.5, delta=0.5)},
        }
        _write_recency_results(tmp_path, families)
        bank = env.Bank(path=bank_path)
        judges = [env.ReplayJudge("ok", data_dir=str(tmp_path)),
                  env.ReplayJudge("bad", data_dir=str(tmp_path))]
        e = env.Environment(judges, bank=bank, budget=10, seed=0,
                             log_path=str(tmp_path / "log.jsonl"))
        o = e.observe()
        assert o["judges_valid"] == ["ok"]
        assert list(o["judges_excluded"]) == ["bad"]
        assert o["judges_excluded"]["bad"] == 9.0

    def test_step_refuses_to_probe_an_excluded_judge(self, tmp_path):
        act_texts = ["an act"]
        bank_path = _write_kernel_payload(tmp_path, act_texts)
        _write_recency_results(tmp_path, {
            "bad": {"model": "m", "acts": _acts_block(act_texts, base=9.0, rewritten=9.5, delta=0.5)},
        })
        bank = env.Bank(path=bank_path)
        judge = env.ReplayJudge("bad", data_dir=str(tmp_path))
        e = env.Environment([judge], bank=bank, budget=10, seed=0,
                             log_path=str(tmp_path / "log.jsonl"))
        with pytest.raises(ValueError):
            e.step("bad", "euphemism", n=1)


# --------------------------------------------------------------- overlap floor

WORDS20 = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
           "kilo lima mike november oscar papa quebec romeo sierra tango").split()
FILLER = ("zulu yankee whiskey uniform victor xtable ybridge zcanyon wharf "
          "quill spool crane otter badger ferret").split()

assert len(WORDS20) == 20  # the boundary arithmetic below (3/20, 2/20) depends on this


class TestOverlapFloor:
    """Defends against: the semantic-preservation floor silently accepting a
    rewrite that drifted too far from the source act, or rejecting one that
    didn't."""

    def test_exactly_at_floor_is_not_rejected(self):
        act = " ".join(WORDS20)
        rewrite = " ".join(WORDS20[:3] + FILLER)  # 3/20 = 0.15 == OVERLAP_FLOOR
        ov = env.overlap(act, rewrite)
        assert ov == pytest.approx(OVERLAP_FLOOR)
        assert not (ov < OVERLAP_FLOOR)  # step()'s actual rejection test

    def test_just_below_floor_is_rejected(self):
        act = " ".join(WORDS20)
        rewrite = " ".join(WORDS20[:2] + FILLER)  # 2/20 = 0.10 < 0.15
        ov = env.overlap(act, rewrite)
        assert ov < OVERLAP_FLOOR

    def test_well_above_floor_passes(self):
        act = " ".join(WORDS20)
        rewrite = " ".join(WORDS20[:15] + FILLER)  # 15/20 = 0.75
        ov = env.overlap(act, rewrite)
        assert ov > OVERLAP_FLOOR

    def test_step_rejects_low_overlap_rewrite(self, tmp_path):
        act_text = " ".join(WORDS20)
        rewrite_text = " ".join(WORDS20[:2] + FILLER)  # below floor
        p = tmp_path / "kernel_payload.json"
        p.write_text(json.dumps({"exp1": [
            {"act": act_text, "moves": {m: rewrite_text for m in env.MOVES}}
        ]}))
        _write_recency_results(tmp_path, {
            "j": {"model": "m", "acts": _acts_block([act_text], base=2.0, rewritten=6.0, delta=1.0)}
        })
        bank = env.Bank(path=str(p))
        judge = env.ReplayJudge("j", data_dir=str(tmp_path))
        e = env.Environment([judge], bank=bank, budget=10, seed=0,
                             log_path=str(tmp_path / "log.jsonl"))
        fb = e.step("j", "euphemism", n=1)
        assert fb["scored"] == 0
        assert fb["rejected"] == 1

    def test_step_accepts_high_overlap_rewrite(self, tmp_path):
        act_text = " ".join(WORDS20)
        rewrite_text = " ".join(WORDS20[:15] + FILLER)  # above floor
        p = tmp_path / "kernel_payload.json"
        p.write_text(json.dumps({"exp1": [
            {"act": act_text, "moves": {m: rewrite_text for m in env.MOVES}}
        ]}))
        _write_recency_results(tmp_path, {
            "j": {"model": "m", "acts": _acts_block([act_text], base=2.0, rewritten=6.0, delta=1.0)}
        })
        bank = env.Bank(path=str(p))
        judge = env.ReplayJudge("j", data_dir=str(tmp_path))
        e = env.Environment([judge], bank=bank, budget=10, seed=0,
                             log_path=str(tmp_path / "log.jsonl"))
        fb = e.step("j", "euphemism", n=1)
        assert fb["scored"] == 1
        assert fb["rejected"] == 0


# ------------------------------------------------- greedy_agent round-robin depth

class TestGreedyAgentDepthPhaseRoundRobin:
    """Regression test for the exploitation bug found and fixed earlier this
    session: an earlier version of greedy_agent always deepened
    judges_valid[0] once breadth was exhausted, and never came back to the
    other valid judges. If that regressed, the counts asserted below would
    collapse onto one judge instead of splitting evenly."""

    def test_depth_phase_cycles_through_all_valid_judges_evenly(
        self, three_valid_judges_bank_and_judges, tmp_path
    ):
        bank, judges = three_valid_judges_bank_and_judges
        e = env.Environment(judges, bank=bank, budget=1000, seed=0,
                             log_path=str(tmp_path / "log.jsonl"))
        n_breadth_cells = len(judges) * len(env.MOVES)  # 3 judges x 6 moves = 18
        depth_rounds = 12
        trace = env.greedy_agent(e, rounds=n_breadth_cells + depth_rounds)

        breadth, depth = trace[:n_breadth_cells], trace[n_breadth_cells:]
        assert len(depth) == depth_rounds
        assert all(t["why"].startswith("breadth") for t in breadth)
        assert all(t["why"].startswith("depth") for t in depth)

        counts = Counter(t["judge"] for t in depth)
        assert set(counts) == {"j0", "j1", "j2"}
        # 12 depth rounds over 3 judges divides evenly -- exact 4/4/4 is the
        # round-robin signature; the old bug would have produced {j0: 12}.
        assert counts == {"j0": 4, "j1": 4, "j2": 4}


# ------------------------------------------------ binary_verdict_pair wiring bug

class TestBinaryVerdictPairMissingBinField:
    """Regression test for the dead binary_verdict_pair field: no per-operator
    'moves' entry in the real data ever has a 'bin' key, so the second
    element of binary_pair was silently always
    None. This must stay explicitly None (a known gap), never coerce to a
    falsy-looking default like False or 0 that would misrepresent an absent
    judge verdict as a real one."""

    def test_score_second_element_is_none_when_bin_key_absent(self, tmp_path):
        judge = make_replay_judge(
            tmp_path, "binfam",
            {"act x": {"base": 2.0, "base_bin": 1,
                       "moves": {"agent_deletion": {"score": 6.0, "delta": 4.0}}}},
        )
        result = judge.score("act x", "agent_deletion")
        assert result["binary_pair"] == (1, None)

    def test_step_logs_none_second_element_through_the_real_path(
        self, make_single_judge_env
    ):
        e = make_single_judge_env("jbin", ["an act about something"],
                                   base=2.0, rewritten=6.0, delta=4.0,
                                   include_bin=False, base_bin=1)
        fb = e.step("jbin", "euphemism", n=1)
        assert fb["scored"] == 1
        with open(e.log_path) as f:
            records = [json.loads(line) for line in f]
        [scored] = [r for r in records if r["validator_verdict"] == "pass"]
        base_bin, rewrite_bin = scored["binary_verdict_pair"]
        assert base_bin == 1
        assert rewrite_bin is None


# ---------------------------------------------------------------- threshold_flip

class TestThresholdFlip:
    """Defends against threshold_flip repeating the binary_verdict_pair
    mistake: it must return None (not a default boolean) when either score
    is missing, and must key strictly off literal <= GATE and rewritten >
    GATE -- tested at the exact gate boundary, not just far from it."""

    @pytest.mark.parametrize(
        "base, rewritten, expected, case_id",
        [
            (None, 5.0, None, "missing_literal_score"),
            (3.0, None, None, "missing_rewritten_score"),
            (None, None, None, "both_missing"),
            (GATE, GATE, False, "rewritten_equal_to_gate_is_not_a_flip"),
            (GATE, GATE + 0.01, True, "rewritten_just_above_gate_flips"),
            (GATE - 0.01, GATE, False, "rewritten_at_gate_not_strictly_above"),
            (GATE - 0.01, GATE + 0.01, True, "clean_crossing_both_sides_of_gate"),
            (GATE + 1.0, GATE + 2.0, False, "literal_already_above_gate_never_flips"),
        ],
    )
    def test_threshold_flip_value(self, tmp_path, base, rewritten, expected, case_id):
        acts = {
            "the test act": {
                "base": base,
                "base_bin": None,
                "moves": {"euphemism": {"score": rewritten, "delta": None}},
            }
        }
        judge = make_replay_judge(tmp_path, f"flip_{case_id}", acts)
        result = judge.score("the test act", "euphemism")
        assert result["threshold_flip"] is expected, case_id

    def test_step_counts_a_flip_when_gate_is_crossed(self, make_single_judge_env):
        e = make_single_judge_env("jflip", ["act for a flip"],
                                   base=2.0, rewritten=6.0, delta=4.0)
        fb = e.step("jflip", "euphemism", n=1)
        assert fb["scored"] == 1
        assert fb["threshold_flips"] == 1

    def test_step_does_not_count_a_flip_when_gate_is_not_crossed(self, make_single_judge_env):
        e = make_single_judge_env("jnoflip", ["act with no flip"],
                                   base=2.0, rewritten=3.0, delta=1.0)
        fb = e.step("jnoflip", "euphemism", n=1)
        assert fb["scored"] == 1
        assert fb["threshold_flips"] == 0


# --------------------------------------------------------- random_agent + env.rng

class TestRandomAgentUsesEnvironmentRNG:
    """Defends against random_agent quietly depending on the global `random`
    module instead of env.rng: that would break the reproducibility
    guarantee this environment is built around (same seed -> same
    trajectory, stated in env.py's own module docstring)."""

    def _build_env(self, tmp_path, suffix):
        bank_path = _write_kernel_payload(tmp_path, SIX_ACT_TEXTS)
        families = {
            jk: {"model": f"{jk}-model",
                 "acts": _acts_block(SIX_ACT_TEXTS, base=2.0, rewritten=6.0, delta=1.0)}
            for jk in ("j0", "j1", "j2")
        }
        _write_recency_results(tmp_path, families)
        bank = env.Bank(path=bank_path)
        judges = [env.ReplayJudge(jk, data_dir=str(tmp_path)) for jk in ("j0", "j1", "j2")]
        return env.Environment(judges, bank=bank, budget=1000, seed=7,
                                log_path=str(tmp_path / f"log_{suffix}.jsonl"))

    def test_same_seed_gives_same_pick_sequence_even_if_global_random_state_differs(
        self, tmp_path
    ):
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        e1 = self._build_env(dir_a, "a")
        random.seed(1)
        [random.random() for _ in range(50)]  # perturb the global module's state
        trace1 = env.random_agent(e1, rounds=10)

        e2 = self._build_env(dir_b, "b")
        random.seed(999999)
        [random.random() for _ in range(7)]  # perturb it differently
        trace2 = env.random_agent(e2, rounds=10)

        picks1 = [(t["judge"], t["move"]) for t in trace1]
        picks2 = [(t["judge"], t["move"]) for t in trace2]
        assert picks1 == picks2

    def test_random_agent_never_calls_the_global_random_module(self, tmp_path, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("random_agent must use env.rng, not the global random module")

        monkeypatch.setattr(env.random, "choice", _boom)
        monkeypatch.setattr(env.random, "random", _boom)
        e = self._build_env(tmp_path, "guarded")
        trace = env.random_agent(e, rounds=6)
        assert len(trace) == 6


# ------------------------------------------------------------------- ucb_agent

def _build_three_judge_env(tmp_path, suffix, budget=1000, seed=7):
    """Three instrument-valid judges over the same six-act synthetic bank:
    3 x 6 = 18 UCB arms, a count the initialization tests below depend on."""
    bank_path = _write_kernel_payload(tmp_path, SIX_ACT_TEXTS)
    families = {
        jk: {"model": f"{jk}-model",
             "acts": _acts_block(SIX_ACT_TEXTS, base=2.0, rewritten=6.0, delta=1.0)}
        for jk in ("j0", "j1", "j2")
    }
    _write_recency_results(tmp_path, families)
    bank = env.Bank(path=bank_path)
    judges = [env.ReplayJudge(jk, data_dir=str(tmp_path)) for jk in ("j0", "j1", "j2")]
    return env.Environment(judges, bank=bank, budget=budget, seed=seed,
                            log_path=str(tmp_path / f"log_{suffix}.jsonl"))


class TestUCBAgentUsesEnvironmentRNG:
    """Same defence TestRandomAgentUsesEnvironmentRNG makes, for the third
    policy: ucb_agent breaks ties with env.rng, and if it quietly reached for
    the global `random` module instead, the environment's reproducibility
    guarantee (same seed -> same trajectory) would silently stop holding for
    this policy while still holding for the other two -- which is exactly the
    kind of asymmetry that would invalidate a three-policy comparison."""

    def test_same_seed_gives_same_pick_sequence_even_if_global_random_state_differs(
        self, tmp_path
    ):
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        e1 = _build_three_judge_env(dir_a, "a")
        random.seed(1)
        [random.random() for _ in range(50)]  # perturb the global module's state
        trace1 = env.ucb_agent(e1, rounds=30)

        e2 = _build_three_judge_env(dir_b, "b")
        random.seed(999999)
        [random.random() for _ in range(7)]  # perturb it differently
        trace2 = env.ucb_agent(e2, rounds=30)

        picks1 = [(t["judge"], t["move"]) for t in trace1]
        picks2 = [(t["judge"], t["move"]) for t in trace2]
        assert picks1 == picks2

    def test_ucb_agent_never_calls_the_global_random_module(self, tmp_path, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("ucb_agent must use env.rng, not the global random module")

        monkeypatch.setattr(env.random, "choice", _boom)
        monkeypatch.setattr(env.random, "random", _boom)
        e = _build_three_judge_env(tmp_path, "guarded")
        trace = env.ucb_agent(e, rounds=25)
        assert len(trace) == 25


class TestUCBAgentInitialization:
    """UCB1's regret bound assumes every arm has been pulled at least once
    before the mean+bonus rule starts choosing; an implementation that let the
    bonus term run with a zero pull count (or that skipped straight to the
    argmax) would divide by zero or, worse, lock onto whichever arm happened to
    be sampled first. This pins the initialization sweep: all 18 arms, each
    exactly once, before any arm is pulled a second time."""

    def test_every_arm_is_pulled_once_before_any_arm_is_pulled_twice(self, tmp_path):
        e = _build_three_judge_env(tmp_path, "init")
        n_arms = 3 * len(env.MOVES)  # 3 judges x 6 operators = 18
        trace = env.ucb_agent(e, rounds=n_arms + 10)

        init, after = trace[:n_arms], trace[n_arms:]
        assert len(after) == 10
        assert all(t["why"].startswith("ucb init") for t in init)
        assert all(t["why"].startswith("ucb:") for t in after)

        picks = [(t["judge"], t["move"]) for t in init]
        assert len(set(picks)) == n_arms          # no arm pulled twice during init
        assert all(t["arm_pulls"] == 1 for t in init)

    def test_init_phase_covers_every_judge_operator_pair_exactly(self, tmp_path):
        e = _build_three_judge_env(tmp_path, "cover")
        n_arms = 3 * len(env.MOVES)
        trace = env.ucb_agent(e, rounds=n_arms)
        expected = {(jk, m) for jk in ("j0", "j1", "j2") for m in env.MOVES}
        assert {(t["judge"], t["move"]) for t in trace} == expected


class TestUCBRewardNormalization:
    """UCB1's exploration bonus is only calibrated against rewards in [0, 1].
    If normalize_delta ever returned a value outside that range the bonus would
    be silently mis-scaled -- the arm would look better or worse than any
    bonus could offset -- so the bounds are pinned at the extremes the frozen
    data actually reaches, not just near the middle."""

    @pytest.mark.parametrize("delta, expected", [
        (env.DELTA_MIN, 0.0),
        (env.DELTA_MAX, 1.0),
        (0.0, 0.5),
    ])
    def test_endpoints_and_midpoint(self, delta, expected):
        assert env.normalize_delta(delta) == pytest.approx(expected)

    def test_is_order_preserving(self):
        # The environment ranks operators by RAW mean delta (Environment._gap).
        # A non-monotone map (e.g. clipping negatives) could make the policy
        # optimise a different ranking than the one the environment reports.
        deltas = [-10.0, -3.5, -0.1, 0.0, 0.1, 3.5, 10.0]
        rewards = [env.normalize_delta(d) for d in deltas]
        assert rewards == sorted(rewards)
        assert all(0.0 <= r <= 1.0 for r in rewards)


class TestUCBAgentHandlesCellsThatNeverScore:
    """A cell whose rewrites all fail the overlap floor spends budget and
    returns no delta, so its SCORED count stays at zero forever. An agent that
    drove its initialization off observe()["unprobed"] -- which counts scored
    observations -- would keep re-selecting that cell and never leave the init
    phase. ucb_agent must count the pull, not the score."""

    def test_a_permanently_rejecting_cell_does_not_stall_the_agent(self, tmp_path):
        act_texts = [" ".join(WORDS20 + [f"unique{i}"]) for i in range(4)]
        dead_move = env.MOVES[0]
        # every operator's rewrite is the act text itself (overlap 1.0) except
        # dead_move's, which keeps only 2 of the >=20 content words -> below floor
        p = tmp_path / "kernel_payload.json"
        p.write_text(json.dumps({"exp1": [
            {"act": t,
             "moves": {m: (" ".join(WORDS20[:2] + FILLER) if m == dead_move else t)
                       for m in env.MOVES}}
            for t in act_texts
        ]}))
        _write_recency_results(tmp_path, {
            "j": {"model": "m", "acts": _acts_block(act_texts, base=2.0,
                                                     rewritten=6.0, delta=1.0)}
        })
        bank = env.Bank(path=str(p))
        judge = env.ReplayJudge("j", data_dir=str(tmp_path))
        e = env.Environment([judge], bank=bank, budget=500, seed=3,
                             log_path=str(tmp_path / "dead.jsonl"))

        trace = env.ucb_agent(e, rounds=40)

        assert len(trace) == 40                       # ran to completion, no stall
        dead = [t for t in trace if t["move"] == dead_move]
        assert dead, "the permanently-rejecting cell must still be pulled at least once"
        assert all(t["scored"] == 0 and t["rejected"] > 0 for t in dead)
        assert e.results.get(("j", dead_move)) is None  # never accumulated a delta
        # and the init phase still ended: later rounds use the mean+bonus rule
        assert any(t["why"].startswith("ucb:") for t in trace)
