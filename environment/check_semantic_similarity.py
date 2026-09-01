#!/usr/bin/env python3
"""A second, independent semantic-preservation check, to cross-validate the
overlap floor already enforced in env.py -- not to replace it.

WHY THIS EXISTS: env.py's only semantic-preservation signal is `overlap()`,
content-word set overlap between the literal act and a rewrite, gated at
`OVERLAP_FLOOR = 0.15`. That is ONE method (lexical overlap) from ONE family
(bag-of-words). A live judge-panel defense question this environment should
be able to answer without hand-waving is "how do you know your rewrite
operators are actually semantically equivalent to the original act, and not
just that they share some words?" A single heuristic answering that about
itself is weak evidence. What strengthens it is an INDEPENDENT signal --
different method, different failure mode -- that either agrees with the
existing floor (two methods converging is stronger evidence than one) or
disagrees (which would be real information about where the overlap floor is
wrong, not something to hide).

This script is that second signal: sentence-embedding cosine similarity
(`sentence-transformers/all-MiniLM-L6-v2`, a small, free, CPU-only,
well-established model -- no GPU, no API key, no cost) computed for every
(literal act, rewrite) pair across all six operators in the frozen act bank.
Embedding similarity captures paraphrase-level semantic closeness; word
overlap captures literal lexical retention. They can and do diverge -- a
fluent paraphrase can preserve meaning while dropping most content words
(agent-deletion, nominalization are built to do exactly that), which is
itself informative about what OVERLAP_FLOOR is and is not measuring.

WHAT THIS SCRIPT DOES NOT DO, on purpose:
  - It does NOT change OVERLAP_FLOOR, `overlap()`, or any accept/reject logic
    in env.py. The campaign already run (campaign_summary.json, the
    exploration_log*.jsonl files) used the existing single-signal gate; that
    result must not be invalidated by silently changing what "passes" means
    after the fact.
  - It does NOT introduce a new embedding-based gate or threshold that
    filters anything. It reports descriptive statistics only: per-operator
    mean/median cosine similarity, and how that distribution relates to the
    existing overlap-floor pass/fail split, using each operator's OWN
    passing- and rejected-group statistics as the comparison points rather
    than an invented cutoff.
  - It imports `Bank`, `MOVES`, `overlap`, and `OVERLAP_FLOOR` directly from
    env.py rather than reimplementing them, so "which pairs pass the
    existing floor" is computed by the actual production code, not a
    hand-copied approximation of it.

Run once, offline after the model is cached:

    python3 environment/check_semantic_similarity.py

First run downloads the ~90MB model from Hugging Face (network required
once; cached under ~/.cache/huggingface after that). Writes
environment/semantic_similarity_report.json with the real numbers.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import Bank, MOVES, overlap, OVERLAP_FLOOR  # noqa: E402  (reuse the real gate logic)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = os.path.join(HERE, "semantic_similarity_report.json")


def _stats(values):
    import statistics as st
    values = list(values)
    if not values:
        return None
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "mean": round(st.mean(values), 4),
        "median": round(st.median(values), 4),
        "max": round(max(values), 4),
    }


def _pearson_r(xs, ys):
    import numpy as np
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return round(float(np.corrcoef(xs, ys)[0, 1]), 4)


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    bank = Bank()
    n_acts = len(bank)
    print(f"Loaded {n_acts} complete acts x {len(MOVES)} operators from the frozen bank.")

    print(f"Loading {MODEL_NAME} (CPU, no GPU needed; first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    act_texts = [a["act"] for a in bank.acts]
    act_emb = np.asarray(model.encode(act_texts, normalize_embeddings=True,
                                       show_progress_bar=False))

    move_emb = {}
    for m in MOVES:
        texts = [a["moves"][m] for a in bank.acts]
        move_emb[m] = np.asarray(model.encode(texts, normalize_embeddings=True,
                                               show_progress_bar=False))

    # Per-pair rows: one per (act, operator), 81 x 6 = 486 total.
    rows = []
    for i, a in enumerate(bank.acts):
        for m in MOVES:
            ov = overlap(a["act"], a["moves"][m])
            # embeddings are L2-normalized, so dot product == cosine similarity
            cos = float(np.dot(act_emb[i], move_emb[m][i]))
            rows.append({
                "act_id": a["act_id"],
                "operator": m,
                "act": a["act"],
                "rewrite": a["moves"][m],
                "overlap": round(ov, 4),
                "cosine": round(cos, 4),
                "passes_overlap_floor": ov >= OVERLAP_FLOOR,
            })

    per_operator = {}
    for m in MOVES:
        mrows = [r for r in rows if r["operator"] == m]
        passing = [r for r in mrows if r["passes_overlap_floor"]]
        rejected = [r for r in mrows if not r["passes_overlap_floor"]]

        cos_all = [r["cosine"] for r in mrows]
        ov_all = [r["overlap"] for r in mrows]
        cos_pass = [r["cosine"] for r in passing]
        cos_rej = [r["cosine"] for r in rejected]

        passing_mean = (sum(cos_pass) / len(cos_pass)) if cos_pass else None
        rejected_mean = (sum(cos_rej) / len(cos_rej)) if cos_rej else None

        # Disagreement counts, anchored to each group's OWN mean (no invented
        # global threshold): a "passing" pair whose embedding similarity sits
        # below the typical REJECTED pair's similarity is a case where the two
        # signals disagree about that pair; symmetric for rejected pairs.
        n_passing_below_rejected_mean = (
            sum(1 for c in cos_pass if c < rejected_mean) if rejected_mean is not None else None
        )
        n_rejected_above_passing_mean = (
            sum(1 for c in cos_rej if c > passing_mean) if passing_mean is not None else None
        )

        worst_passing = min(passing, key=lambda r: r["cosine"]) if passing else None
        best_rejected = max(rejected, key=lambda r: r["cosine"]) if rejected else None

        per_operator[m] = {
            "n": len(mrows),
            "overlap_stats": _stats(ov_all),
            "cosine_stats": _stats(cos_all),
            "pearson_r_overlap_vs_cosine": _pearson_r(ov_all, cos_all),
            "n_pass_overlap_floor": len(passing),
            "n_reject_overlap_floor": len(rejected),
            "reject_rate": round(len(rejected) / len(mrows), 4) if mrows else None,
            "cosine_stats_passing_group": _stats(cos_pass),
            "cosine_stats_rejected_group": _stats(cos_rej),
            "n_passing_pairs_below_rejected_groups_mean_cosine": n_passing_below_rejected_mean,
            "n_rejected_pairs_above_passing_groups_mean_cosine": n_rejected_above_passing_mean,
            "worked_example_lowest_cosine_among_passing": (
                None if worst_passing is None else {
                    "act_id": worst_passing["act_id"], "overlap": worst_passing["overlap"],
                    "cosine": worst_passing["cosine"], "act": worst_passing["act"],
                    "rewrite": worst_passing["rewrite"],
                }
            ),
            "worked_example_highest_cosine_among_rejected": (
                None if best_rejected is None else {
                    "act_id": best_rejected["act_id"], "overlap": best_rejected["overlap"],
                    "cosine": best_rejected["cosine"], "act": best_rejected["act"],
                    "rewrite": best_rejected["rewrite"],
                }
            ),
        }

    all_cos = [r["cosine"] for r in rows]
    all_ov = [r["overlap"] for r in rows]
    all_pass = [r for r in rows if r["passes_overlap_floor"]]
    all_rej = [r for r in rows if not r["passes_overlap_floor"]]

    overall = {
        "n_pairs": len(rows),
        "overlap_stats": _stats(all_ov),
        "cosine_stats": _stats(all_cos),
        "pearson_r_overlap_vs_cosine": _pearson_r(all_ov, all_cos),
        "n_pass_overlap_floor": len(all_pass),
        "n_reject_overlap_floor": len(all_rej),
        "reject_rate": round(len(all_rej) / len(rows), 4) if rows else None,
    }

    report = {
        "purpose": ("Independent cross-check of the existing OVERLAP_FLOOR "
                    "semantic-preservation gate in env.py, via sentence-embedding "
                    "cosine similarity. Descriptive statistics only -- computing "
                    "this does not change which rewrites pass into the campaign; "
                    "see the module docstring in this file for what was and was "
                    "not modified."),
        "model": MODEL_NAME,
        "embedding_dim": int(act_emb.shape[1]),
        "n_acts": n_acts,
        "n_operators": len(MOVES),
        "operators": list(MOVES),
        "overlap_floor": OVERLAP_FLOOR,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compute_seconds": round(time.time() - t0, 1),
        "per_operator": per_operator,
        "overall": overall,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {OUT_PATH}")
    print("\nPer-operator cosine similarity (mean / median), overlap reject rate:")
    for m in MOVES:
        po = per_operator[m]
        cs = po["cosine_stats"]
        print(f"  {m:18} cosine mean={cs['mean']:.3f} median={cs['median']:.3f}  "
              f"overlap_reject_rate={po['reject_rate']:.3f}  "
              f"r(overlap,cosine)={po['pearson_r_overlap_vs_cosine']}")
    print(f"\nOverall: n_pairs={overall['n_pairs']} "
          f"cosine mean={overall['cosine_stats']['mean']:.3f} "
          f"r(overlap,cosine)={overall['pearson_r_overlap_vs_cosine']}")


if __name__ == "__main__":
    main()
