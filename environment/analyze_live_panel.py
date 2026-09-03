"""Analyse a live multi-model panel run against the environment's own null model.

Reads the per-model reports written by run_live_demo.py and answers, for each
model that cleared the instrument-validity gate, two questions fixed in advance
in CHINESE_PANEL_PREREGISTRATION.md:

  H1  does euphemism have the highest mean delta of the six operators?
  H2  does the rank1 minus rank2 gap exceed the p95 of the environment's own
      null model?

The null model is the same one the frozen campaign uses: shuffle the observed
deltas across operators while HOLDING each operator's realised sample size, then
rebuild the rank1 minus rank2 gap. Holding realised n is the whole point. A gap
built from n=194 on one operator and n=1 on five is not comparable to a gap
built from six operators at n=40, and conditioning on the realised allocation is
what makes the comparison honest.

This reports every model that cleared the gate, including models where euphemism
does NOT rank first, every model the gate excluded, and every model that errored.
That is a commitment made in the pre-registration before any data existed.

Offline. Reads JSON that already exists; makes no API call and needs no key.
"""
import argparse, collections, glob, json, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def null_gap_p95(per_operator_deltas, trials=2000, seed=0):
    """p95 of the rank1-minus-rank2 gap under random reassignment.

    Sample sizes are held at their realised values, so the only thing being
    destroyed is which operator a delta belonged to.
    """
    sizes = [(op, len(v)) for op, v in per_operator_deltas.items()]
    pool = [d for v in per_operator_deltas.values() for d in v]
    if len(sizes) < 2 or len(pool) < 2:
        return None
    rng = random.Random(seed)
    gaps = []
    for _ in range(trials):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        i, means = 0, []
        for _op, n in sizes:
            means.append(st.mean(shuffled[i:i + n]))
            i += n
        means.sort(reverse=True)
        gaps.append(means[0] - means[1])
    gaps.sort()
    return gaps[int(0.95 * len(gaps))]


def analyse(report_paths, trials=2000, seed=0):
    out = {"models": [], "trials": trials, "seed": seed}
    for path in sorted(report_paths):
        blob = json.load(open(path))
        for m in (blob.get("results") or blob.get("models") or []):
            row = {"model": m.get("model"), "provider": m.get("provider"),
                   "policy": m.get("policy"), "error": m.get("error"),
                   "literal_mean": m.get("literal_mean"),
                   "instrument_valid": m.get("instrument_valid")}
            probes = m.get("probes") or []
            row["n_probes"] = len(probes)
            if row["error"]:
                row["status"] = "errored"
                out["models"].append(row); continue
            if not m.get("instrument_valid"):
                # Reported, never silently dropped: this is exactly how the two
                # IBM Granite models are handled in the frozen panel.
                row["status"] = "excluded_by_gate"
                out["models"].append(row); continue
            if not probes:
                row["status"] = "no_probes"
                out["models"].append(row); continue

            per = collections.defaultdict(list)
            for p in probes:
                per[p["operator"]].append(p["delta"])
            means = {op: st.mean(v) for op, v in per.items()}
            ranked = sorted(means.items(), key=lambda kv: -kv[1])
            gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
            p95 = null_gap_p95(per, trials=trials, seed=seed)

            row.update({
                "status": "analysed",
                "per_operator": {op: {"mean_delta": round(means[op], 4),
                                      "n": len(per[op])} for op in per},
                "n_min": min(len(v) for v in per.values()),
                "n_max": max(len(v) for v in per.values()),
                "rank1": ranked[0][0], "rank2": ranked[1][0] if len(ranked) > 1 else None,
                "gap": None if gap is None else round(gap, 4),
                "null_gap_p95": None if p95 is None else round(p95, 4),
                "H1_euphemism_rank1": ranked[0][0] == "euphemism",
                "H2_gap_clears_null": (None if (gap is None or p95 is None)
                                       else bool(gap > p95)),
                "threshold_flips": sum(1 for p in probes if p.get("threshold_flip")),
            })
            out["models"].append(row)

    ok = [r for r in out["models"] if r["status"] == "analysed"]
    out["summary"] = {
        "models_reported": len(out["models"]),
        "cleared_gate": len(ok),
        "excluded_by_gate": sum(1 for r in out["models"] if r["status"] == "excluded_by_gate"),
        "errored": sum(1 for r in out["models"] if r["status"] == "errored"),
        "H1_euphemism_rank1": sum(1 for r in ok if r["H1_euphemism_rank1"]),
        "H2_clears_null": sum(1 for r in ok if r["H2_gap_clears_null"]),
        "rank1_counts": dict(collections.Counter(r["rank1"] for r in ok)),
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reports", required=True,
                   help="glob for the per-model report JSON files")
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "live_panel", "live_panel_analysis.json"),
                   help="default writes beside the reports it summarises, so a rerun "
                        "updates the committed artifact instead of dropping a second "
                        "copy one directory up (default: %(default)s)")
    a = p.parse_args()

    res = analyse(glob.glob(a.reports), trials=a.trials, seed=a.seed)
    hdr = (f"  {'model':<34}{'lit':>6}{'gate':>6}{'rank1':<19}"
           f"{'gap':>8}{'null p95':>10}{'clears':>8}{'n':>10}")
    print("=" * len(hdr))
    print("  LIVE PANEL vs THE ENVIRONMENT'S OWN NULL MODEL")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in res["models"]:
        if r["status"] != "analysed":
            print(f"  {r['model']:<34}{'-':>6}{r['status']:>6}  {r['status']}")
            continue
        print(f"  {r['model']:<34}{r['literal_mean']:>6.2f}{'yes':>6}{r['rank1']:<19}"
              f"{r['gap']:>8.3f}{r['null_gap_p95']:>10.3f}"
              f"{('YES' if r['H2_gap_clears_null'] else 'no'):>8}"
              f"{str(r['n_min']) + '-' + str(r['n_max']):>10}")
    s = res["summary"]
    print()
    print(f"  {s['cleared_gate']} of {s['models_reported']} models cleared the validity gate "
          f"({s['excluded_by_gate']} excluded, {s['errored']} errored)")
    print(f"  euphemism ranked first in {s['H1_euphemism_rank1']} of {s['cleared_gate']}")
    print(f"  gap cleared its own null in {s['H2_clears_null']} of {s['cleared_gate']}")
    print(f"  rank-1 operators observed: {s['rank1_counts']}")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
