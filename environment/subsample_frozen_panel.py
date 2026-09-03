#!/usr/bin/env python3
"""Does the live panels' weaker euphemism ranking come from smaller samples?

The frozen panel scores every operator on the full act bank (N = 68 to 81 per
operator). The live panels run under a probe budget, so each operator lands at
n = 27 to 47. Euphemism ranks first in 9 of 9 frozen cells but only around
second in the live panels, and the obvious alternative explanation is that the
live runs are simply underpowered: at small n, several operators sit within
noise of one another and which one comes out on top is close to a coin flip.

This script tests that directly and offline. For each frozen judge it draws
`--resamples` subsamples of size n WITHOUT replacement from the real per-act
deltas, recomputes the operator ranking each time, and reports how often
euphemism still comes first and where it lands on average.

If the frozen judges also collapse toward rank 2 at n = 30, then sample size
explains the live results and no claim about model generation survives. If they
hold at rank 1, sample size is ruled out.

Needs no API key and no network; it reads the committed frozen data only.
"""
import argparse, json, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE = 4.0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=os.path.join(ROOT, "data", "recency_results.json"))
    p.add_argument("--sizes", default="30,40,50", help="subsample sizes to test")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "subsample_frozen_panel.json"))
    a = p.parse_args()

    random.seed(a.seed)
    sizes = [int(s) for s in a.sizes.split(",")]
    data = json.load(open(a.data))
    rows = []

    for judge, jd in data.items():
        acts = jd["acts"]
        lits = [x["base"] for x in acts.values() if x.get("base") is not None]
        if not lits:
            # A family whose literal scores were all unparseable cannot be gated,
            # so it cannot be ranked either. Falcon-H1R is the real case.
            rows.append({"judge": judge, "literal_mean": None, "gate_valid": None,
                         "note": "no parseable literal scores; excluded upstream"})
            continue
        literal_mean = sum(lits) / len(lits)
        valid = literal_mean <= GATE
        ops = sorted({o for x in acts.values() for o in x["moves"]})
        per = {o: [x["moves"][o]["delta"] for x in acts.values()
                   if o in x["moves"] and x["moves"][o].get("delta") is not None]
               for o in ops}
        per = {k: v for k, v in per.items() if v}
        if "euphemism" not in per or len(per) < 3:
            rows.append({"judge": judge, "literal_mean": literal_mean,
                         "gate_valid": valid, "note": "insufficient operator coverage"})
            continue
        full_n = min(len(v) for v in per.values())
        entry = {"judge": judge, "model": jd.get("model"),
                 "literal_mean": round(literal_mean, 4), "gate_valid": valid,
                 "full_n": full_n, "n_operators": len(per), "by_size": {}}
        for n in sizes + [full_n]:
            if n > full_n:
                continue
            wins, ranks = 0, []
            for _ in range(a.resamples):
                means = {o: st.mean(random.sample(v, n)) for o, v in per.items()}
                order = sorted(means, key=lambda o: -means[o])
                ranks.append(order.index("euphemism") + 1)
                wins += order[0] == "euphemism"
            entry["by_size"][str(n)] = {
                "euphemism_rank1_rate": round(wins / a.resamples, 4),
                "mean_euphemism_rank": round(st.mean(ranks), 4)}
        rows.append(entry)

    valid_rows = [r for r in rows if r.get("gate_valid") and r.get("by_size")]
    summary = {}
    for n in sizes:
        vals = [r["by_size"][str(n)]["mean_euphemism_rank"]
                for r in valid_rows if str(n) in r["by_size"]]
        if vals:
            summary[f"n={n}"] = {
                "judges": len(vals),
                "mean_euphemism_rank_across_valid_judges": round(sum(vals) / len(vals), 4)}

    out = {
        "what": "Frozen panel subsampled to the live panels' per-operator n",
        "why": ("Rules out sample size as the explanation for euphemism ranking "
                "around second in the live panels while ranking first in 9 of 9 "
                "frozen cells. Only gate-valid judges are summarised, because a "
                "judge that fails the instrument-validity gate is not measuring "
                "the thing being compared."),
        "gate": GATE, "resamples": a.resamples, "seed": a.seed,
        "sampling": "without replacement from real per-act deltas",
        "per_judge": rows, "summary_over_gate_valid_judges": summary,
        "live_panel_comparison": {
            "live_chinese_mean_euphemism_rank": 2.33,
            "live_western_mean_euphemism_rank": 1.83,
            "live_per_operator_n_range": "27 to 47"},
    }
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"{'judge':<16}{'lit':>7}{'gate':>6}  " +
          "".join(f"n={n:<10}" for n in sizes) + "full")
    for r in rows:
        if not r.get("by_size"):
            print(f"{r['judge']:<16}{'--':>7}{'--':>6}  {r.get('note','')}")
            continue
        line = f"{r['judge']:<16}{r['literal_mean']:>7.2f}{'yes' if r['gate_valid'] else 'NO':>6}  "
        for n in sizes + [r["full_n"]]:
            b = r["by_size"].get(str(n))
            line += (f"{b['euphemism_rank1_rate']:.0%}/{b['mean_euphemism_rank']:.2f}".ljust(12)
                     if b else " " * 12)
        print(line)
    print()
    for k, v in summary.items():
        print(f"  gate-valid judges at {k}: mean euphemism rank "
              f"{v['mean_euphemism_rank_across_valid_judges']} over {v['judges']} judges")
    print(f"  live panels for comparison: Chinese 2.33, Western 1.83, at n=27 to 47")
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
