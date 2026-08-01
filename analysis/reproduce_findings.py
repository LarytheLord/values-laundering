#!/usr/bin/env python3
"""Recompute the three headline findings straight from the judged result files
in ../results, so anyone can verify the numbers in the README rather than trust
the stored summaries. Run: python analysis/reproduce_findings.py
"""
import json, os, statistics as st

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
load = lambda n: json.load(open(os.path.join(R, n)))


def finding1():
    print("=" * 72)
    print("Finding 1  euphemism dominates the 6 CDA moves (cross-family judge, N=81)")
    print("=" * 72)
    d = load("crossfamily_bigN_gemma.json"); rows = d["rows"]
    moves = ["agent_deletion", "nominalization", "functionalization",
             "euphemism", "necessity", "aggregation"]
    board = {m: st.mean([r["moves"][m]["delta"] for r in rows if m in r["moves"]]) for m in moves}
    top = max(board, key=board.get)
    for m in sorted(board, key=board.get, reverse=True):
        print(f"  {m:16s} {board[m]:+.3f}" + ("   <- #1" if m == top else ""))
    print(f"  judge = {d['summary']['judge']}, n_acts = {len(rows)}")


def finding2():
    print("\n" + "=" * 72)
    print("Finding 2  it's a real judgment shift, not a rating-scale artifact")
    print("=" * 72)
    rows = load("instrument_robustness_results.json")["rows"]
    num = [r["numeric"]["euph"] - r["numeric"]["base"] for r in rows]
    condemned = [r for r in rows if r["binary"]["base"] == 1]
    released = [r for r in condemned if r["binary"]["euph"] == 0]
    print(f"  numeric 0-10 mean acceptability gain from euphemism: {st.mean(num):+.3f}")
    print(f"  strict binary 'is this morally wrong? yes/no':")
    print(f"    euphemism releases condemnation on {len(released)}/{len(condemned)} "
          f"= {100*len(released)/len(condemned):.0f}% of acts (no scale to nudge)")


def finding3():
    print("\n" + "=" * 72)
    print("Finding 3  the moves don't compound, they saturate")
    print("=" * 72)
    rows = load("compositional_gemma_results.json")["rows"]
    d = lambda r, c: r["scores"][c] - r["scores"]["base"]     # acceptability gain
    for c in ["E", "F", "A", "EF", "EA", "FA", "EFA"]:
        print(f"  mean gain {c:3s} {st.mean([d(r, c) for r in rows]):+.3f}")
    for x, y, xy in [("E", "F", "EF"), ("E", "A", "EA"), ("F", "A", "FA")]:
        inter = st.mean([d(r, xy) - d(r, x) - d(r, y) for r in rows])
        print(f"  interaction I({x},{y}) = {inter:+.3f}   (negative = subadditive)")
    no_better = sum(1 for r in rows
                    if d(r, "EFA") <= max(d(r, "E"), d(r, "F"), d(r, "A")))
    print(f"  stacking all three is no better than the single best move in "
          f"{no_better}/{len(rows)} acts ({round(100*no_better/len(rows))}%)")


if __name__ == "__main__":
    finding1(); finding2(); finding3()
    print("\nAll numbers above are recomputed from results/*.json.")
