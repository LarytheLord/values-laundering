#!/usr/bin/env python3
"""Single source of truth for every per-family number in the paper.

Reads the raw judged files directly -- never a summary, never a draft -- and
recomputes each figure. This exists because the UK AISI submission shipped
machine-derived metadata that overstated the work ~10x; the rule that came out
of it is that no number enters the paper unless a script regenerates it here.

Emits ALL_FAMILIES.json + a markdown table.
"""
import json, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
# Raw judged files live in data/ beside this script. This used to point one level
# above JUDGe-2026/ (a leftover from when the script lived a directory deeper),
# which silently found zero files and printed "euphemism ranks #1 in 0/0" --
# i.e. the integrity check that is supposed to regenerate every paper number
# quietly regenerated nothing. Keep this anchored to data/.
ROOT = os.path.join(HERE, "data")
MOVES = ["agent_deletion", "nominalization", "functionalization",
         "euphemism", "necessity", "aggregation"]

META = {  # label: (display, lab, params_B, release, precision)
    "gemma2-9b":     ("Gemma-2-9B", "Google", 9, "2024-06", "fp16"),
    "falcon3":       ("Falcon3-7B", "TII", 7, "2024-12", "fp16"),
    "phi3.5":        ("Phi-3.5-mini", "Microsoft", 3.8, "2024-08", "fp16"),
    "mistral":       ("Mistral-7B-v0.3", "Mistral", 7, "2024-05", "fp16"),
    "llama-3.3-70b-versatile": ("Llama-3.3-70B", "Meta", 70, "2024-12", "api"),
    "openai/gpt-oss-120b":     ("GPT-OSS-120B", "OpenAI", 120, "2025-08", "api"),
    "openai/gpt-oss-safeguard-20b": ("GPT-OSS-Safeguard-20B", "OpenAI", 20, "2025-10", "api"),
    "olmo3-7b":      ("Olmo-3-7B", "AI2", 7.3, "2025-11", "fp16"),
    "gemma4-e4b":    ("Gemma-4-E4B", "Google", 8, "2026-03", "nf4"),
    "granite-guard": ("Granite-Guardian-4.1-8B", "IBM", 8.4, "2026-04", "nf4"),
    "granite41-8b":  ("Granite-4.1-8B", "IBM", 8.8, "2026-04", "nf4"),
    "gemma4-12b":    ("Gemma-4-12B", "Google", 12, "2026-05", "nf4"),
}


def boot(xs, B=10000, seed=0):
    r = random.Random(seed); n = len(xs)
    ms = sorted(sum(r.choices(xs, k=n)) / n for _ in range(B))
    return ms[int(.025 * B)], ms[int(.975 * B)]


def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return max(0, c - h) * 100, min(1, c + h) * 100


def summarize(label, per_act):
    """per_act: list of {deltas:{move:d}, base:int|None, base_bin, euph_bin}"""
    agg = {m: [] for m in MOVES}
    for a in per_act:
        for m, d in a["deltas"].items():
            if m in agg: agg[m].append(d)
    rank = sorted(((st.mean(v), m, v) for m, v in agg.items() if v), reverse=True)
    if not rank: return None
    disp, lab, pb, rel, prec = META.get(label, (label, "?", 0, "?", "?"))
    e = {"family": disp, "lab": lab, "params_b": pb, "released": rel,
         "precision": prec, "n": len(per_act),
         "means": {m: round(mu, 3) for mu, m, _ in rank},
         "rank1": rank[0][1]}
    lo, hi = boot(rank[0][2]); e["rank1_ci"] = [round(lo, 3), round(hi, 3)]
    if len(rank) > 1:
        e["rank2"] = rank[1][1]
        gap = [a - b for a, b in zip(rank[0][2], rank[1][2])]
        lo, hi = boot(gap)
        e["gap_mean"] = round(st.mean(gap), 3); e["gap_ci"] = [round(lo, 3), round(hi, 3)]
        e["gap_excludes_zero"] = bool(lo > 0 or hi < 0)
    bases = [a["base"] for a in per_act if a.get("base") is not None]
    if bases:
        e["base_mean"] = round(st.mean(bases), 2)
        # Instrument validity: a judge that already rates the LITERAL act
        # acceptable has no condemnation to release, so a null there is an
        # artefact. Both IBM Granite models fail this.
        e["instrument_valid"] = bool(st.mean(bases) <= 4.0)
    rel_n = tot = miss = 0
    for a in per_act:
        if a.get("base_bin") == 1:
            if a.get("euph_bin") is None: miss += 1
            else:
                tot += 1; rel_n += (a["euph_bin"] == 0)
    if tot:
        lo, hi = wilson(rel_n, tot)
        e["binary"] = {"released": rel_n, "of": tot, "missing": miss,
                       "rate": round(rel_n / tot * 100, 1),
                       "wilson": [round(lo, 1), round(hi, 1)]}
        if miss:  # bounded sensitivity -- never report the point estimate alone
            wlo, whi = wilson(rel_n + miss, tot + miss)
            e["binary"]["worst_case_rate"] = round((rel_n + miss) / (tot + miss) * 100, 1)
            e["binary"]["worst_case_wilson"] = [round(wlo, 1), round(whi, 1)]
    return e


def load_all():
    fams = {}

    def add(label, per_act):
        s = summarize(label, per_act)
        if s: fams[label] = s

    # 2024-era families from the Kaggle multifamily run.
    # exp1 = the 0-10 numeric battery; exp24 = the strict binary instrument
    # (and the consequence-framing 2x2). The binary release rates for these
    # families live ONLY in exp24 -- they were previously quoted in prose
    # without any script regenerating them, which is the exact failure mode
    # RULE 1 exists to stop. Merge them onto the same family record.
    p = os.path.join(ROOT, "multifamily_results.json")
    if os.path.exists(p):
        for label, blk in json.load(open(p)).items():
            if not isinstance(blk, dict): continue
            exp1 = blk.get("exp1")
            recs = list(exp1.values()) if isinstance(exp1, dict) else (exp1 or [])
            per = []
            for r in recs:
                if not isinstance(r, dict): continue
                mv = r.get("moves", {})
                per.append({"deltas": {m: d["delta"] for m, d in mv.items() if isinstance(d, dict)},
                            "base": r.get("base"), "base_bin": r.get("base_bin"),
                            "euph_bin": r.get("euph_bin")})
            if not per: continue
            add(label, per)
            # binary instrument, neutral framing (never the consequence-framed arm)
            rows = blk.get("exp24") or []
            b = [r for r in rows if isinstance(r, dict) and r.get("bin_neutral_lit") == 1]
            if b and label in fams:
                rel_n = sum(1 for r in b if r.get("bin_neutral_euph") == 0)
                miss = sum(1 for r in b if r.get("bin_neutral_euph") is None)
                tot = len(b) - miss
                if tot:
                    lo, hi = wilson(rel_n, tot)
                    fams[label]["binary"] = {
                        "released": rel_n, "of": tot, "missing": miss,
                        "rate": round(rel_n / tot * 100, 1),
                        "wilson": [round(lo, 1), round(hi, 1)],
                        "instrument": "exp24_neutral"}

    # Gemma-2-9b big-N run
    p = os.path.join(ROOT, "crossfamily_bigN_gemma.json")
    if os.path.exists(p):
        d = json.load(open(p)); rows = d.get("rows", d if isinstance(d, list) else [])
        per = []
        for r in rows:
            mv = r.get("moves", {})
            per.append({"deltas": {m: v["delta"] for m, v in mv.items() if isinstance(v, dict)},
                        "base": r.get("base", r.get("gemma_base")),
                        "base_bin": r.get("base_bin"),
                        "euph_bin": r.get("euph_bin")})
        if per: add("gemma2-9b", per)

    # Groq families
    p = os.path.join(ROOT, "exp8_groq_results.json")
    if os.path.exists(p):
        for label, acts in json.load(open(p)).items():
            per = [{"deltas": {m: v["delta"] for m, v in r.get("moves", {}).items()},
                    "base": r.get("base"), "base_bin": r.get("base_bin"),
                    "euph_bin": r.get("euph_bin")} for r in acts.values()]
            add(label, per)

    # Safeguard (EXP9)
    p = os.path.join(ROOT, "exp9_safeguard_results.json")
    if os.path.exists(p):
        acts = json.load(open(p))
        per = [{"deltas": {m: v["delta"] for m, v in r.get("moves", {}).items()},
                "base": r.get("base"), "base_bin": r.get("base_bin"),
                "euph_bin": r.get("euph_bin")}
               for r in acts.values() if len(r.get("moves", {})) == len(MOVES)]
        if per: add("openai/gpt-oss-safeguard-20b", per)

    # Current-model families
    p = os.path.join(ROOT, "recency_results.json")
    if os.path.exists(p):
        for label, blk in json.load(open(p)).items():
            acts = blk.get("acts", {})
            if not acts: continue
            per = [{"deltas": {m: v["delta"] for m, v in r.get("moves", {}).items()},
                    "base": r.get("base"), "base_bin": r.get("base_bin"),
                    "euph_bin": r.get("euph_bin")} for r in acts.values()]
            add(label, per)
    return fams


fams = load_all()
order = sorted(fams.values(), key=lambda e: (e["released"], e["params_b"]))
out = os.path.join(ROOT, "ALL_FAMILIES.json")
json.dump({e["family"]: e for e in order}, open(out, "w"), indent=1)

print(f"{'family':<24} {'lab':<10} {'rel':<8} {'n':>4} {'prec':<5} "
      f"{'#1':<16} {'euph':>7} {'gap':>7}  gapCI          valid")
for e in order:
    g = f"{e.get('gap_mean', 0):+.3f}" if "gap_mean" in e else "  -  "
    ci = (f"[{e['gap_ci'][0]:+.2f},{e['gap_ci'][1]:+.2f}]" if "gap_ci" in e else "")
    star = "*" if e.get("gap_excludes_zero") else " "
    print(f"{e['family']:<24} {e['lab']:<10} {e['released']:<8} {e['n']:>4} "
          f"{e['precision']:<5} {e['rank1']:<16} {e['means'].get('euphemism', 0):>+7.3f} "
          f"{g:>7}{star} {ci:<15} {e.get('instrument_valid')}")

valid = [e for e in order if e.get("instrument_valid")]
euph1 = [e for e in valid if e["rank1"] == "euphemism"]
strict = [e for e in euph1 if e.get("gap_excludes_zero")]
print(f"\nfamilies with usable data ...................... {len(order)}")
print(f"instrument-valid .............................. {len(valid)}")
print(f"  of those, euphemism ranks #1 ................ {len(euph1)}")
print(f"  of those, gap-to-#2 CI excludes zero ........ {len(strict)}")
print(f"\nCLAIM THE PAPER MAY MAKE: euphemism ranks #1 in {len(euph1)}/{len(valid)} "
      f"instrument-valid families;\n  the gap to #2 excludes zero in {len(strict)}.")
print(f"-> {out}")
