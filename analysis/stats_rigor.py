#!/usr/bin/env python3
"""Proper uncertainty for the saturation + condemnation-release claims.

Addresses the two statistical attacks a JUDGe reviewer will make:
  1. "Your saturation ordering is point-estimate arithmetic on N=78."
     -> bootstrap CIs over acts + a sign-flip permutation test per interaction term.
  2. "8/21 = 38% is a bare proportion on tiny N."
     -> Wilson score interval.

No model inference. Pure re-analysis of already-judged data.
Run: python analysis/stats_rigor.py
"""
import json, os, random, statistics as st

random.seed(20260803)
# Raw judged files live in JUDGe-2026/data/. Same path bug as consolidate_all.py
# had: pointing at the project root found nothing and crashed on first load.
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Raw judged files are split across two directories: the ten-family replication lives in
# data/, the original three findings in results/. An earlier version looked only in data/,
# so a fresh clone crashed on the first load even though the file was sitting in results/.
# Look in both, and say which paths were tried if it is genuinely missing.
_DIRS = [os.path.join(_BASE, "data"), os.path.join(_BASE, "results")]


def load(n):
    for d in _DIRS:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(f"{n} not found in " + " or ".join(_DIRS))
B = 10000


def boot_ci(vals, f=st.mean, b=B, alpha=0.05):
    n = len(vals)
    reps = sorted(f([vals[random.randrange(n)] for _ in range(n)]) for _ in range(b))
    return f(vals), reps[int((alpha / 2) * b)], reps[int((1 - alpha / 2) * b) - 1]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, c - h, c + h


def perm_p(paired, b=B):
    """H0: the paired effect is 0. Sign-flip permutation on per-act values."""
    obs = abs(st.mean(paired))
    hits = sum(
        abs(st.mean([v if random.random() < .5 else -v for v in paired])) >= obs
        for _ in range(b)
    )
    return (hits + 1) / (b + 1)


def main():
    print("=" * 74)
    print(f"SATURATION - bootstrap 95% CIs over acts (B={B})")
    print("=" * 74)
    rows = load("compositional_gemma_results.json")["rows"]
    g = lambda r, c: r["scores"][c] - r["scores"]["base"]
    for c in ["E", "F", "A", "EF", "EA", "FA", "EFA"]:
        m, lo, hi = boot_ci([g(r, c) for r in rows])
        print(f"  {c:3s} mean gain {m:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\nINTERACTION TERMS  I(X,Y) = d(XY) - d(X) - d(Y)")
    print("  negative => subadditive (saturating)")
    for x, y, xy in [("E", "F", "EF"), ("E", "A", "EA"), ("F", "A", "FA")]:
        paired = [g(r, xy) - g(r, x) - g(r, y) for r in rows]
        m, lo, hi = boot_ci(paired)
        p = perm_p(paired)
        verdict = "SIGNIFICANT" if hi < 0 or lo > 0 else "n.s."
        print(f"  I({x},{y}) = {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  perm p={p:.4f}  [{verdict}]")

    print("\nKEY CLAIM: is stacking all three worse than the best single move?")
    diff = [g(r, "EFA") - max(g(r, "E"), g(r, "F"), g(r, "A")) for r in rows]
    m, lo, hi = boot_ci(diff)
    print(f"  mean(EFA - best_single) = {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  perm p={perm_p(diff):.4f}")
    nb = sum(1 for d in diff if d <= 0)
    p, lo2, hi2 = wilson(nb, len(diff))
    print(f"  stacking no better in {nb}/{len(diff)} acts = {100*p:.1f}%"
          f"  Wilson 95% CI [{100*lo2:.1f}%, {100*hi2:.1f}%]")

    print("\n" + "=" * 74)
    print("CONDEMNATION RELEASE - Wilson score interval")
    print("=" * 74)
    r2 = load("instrument_robustness_results.json")["rows"]
    cond = [r for r in r2 if r["binary"]["base"] == 1]
    rel = [r for r in cond if r["binary"]["euph"] == 0]
    p, lo, hi = wilson(len(rel), len(cond))
    print(f"  euphemism releases condemnation on {len(rel)}/{len(cond)} = {100*p:.1f}%")
    print(f"  Wilson 95% CI [{100*lo:.1f}%, {100*hi:.1f}%]  <- report THIS, not the bare 38%")
    num = [r["numeric"]["euph"] - r["numeric"]["base"] for r in r2]
    m, l, h = boot_ci(num)
    print(f"  numeric 0-10 gain {m:+.3f}  95% CI [{l:+.3f}, {h:+.3f}]")

    print("\n" + "=" * 74)
    print("EUPHEMISM DOMINANCE - is the #1 vs #2 gap real? (N=81)")
    print("=" * 74)
    r1 = load("crossfamily_bigN_gemma.json")["rows"]
    moves = ["agent_deletion", "nominalization", "functionalization",
             "euphemism", "necessity", "aggregation"]
    board = {mv: boot_ci([r["moves"][mv]["delta"] for r in r1 if mv in r["moves"]])
             for mv in moves}
    for mv in sorted(board, key=lambda k: -board[k][0]):
        m, lo, hi = board[mv]
        print(f"  {mv:18s} {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    top, second = sorted(board, key=lambda k: -board[k][0])[:2]
    paired = [r["moves"][top]["delta"] - r["moves"][second]["delta"]
              for r in r1 if top in r["moves"] and second in r["moves"]]
    m, lo, hi = boot_ci(paired)
    print(f"\n  {top} - {second}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  perm p={perm_p(paired):.4f}")
    print("  (CI excluding 0 => the #1 ranking is not sampling noise)")


if __name__ == "__main__":
    main()
