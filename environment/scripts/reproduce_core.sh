#!/usr/bin/env bash
# Thin wrapper around `python3 run_campaign.py`.
#
# What this adds over calling the Python directly: the same environment
# sanity checks as smoke_test.sh (readable error + non-zero exit instead of a
# raw traceback if something's missing), and, after the campaign finishes, a
# printout of each seed's final gap_to_second per judge straight from
# campaign_summary.json -- so a reviewer can compare against the numbers
# documented in README.md/the report without separately opening the JSON.
#
# No environment logic lives here: run_campaign.py runs unmodified, and the
# summary printout only reads back the JSON that script already wrote.
#
# Portable: plain POSIX test + bash builtins only, no bash4+ features
# (no associative arrays, no `mapfile`), works on macOS's default bash 3.2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$ENV_DIR/.." && pwd)"

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

echo "== reproduce_core.sh: environment sanity checks =="

command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH"

[ -f "$ENV_DIR/run_campaign.py" ] || fail "run_campaign.py not found at $ENV_DIR/run_campaign.py"
[ -f "$ENV_DIR/env.py" ] || fail "env.py not found at $ENV_DIR/env.py (run_campaign.py imports it)"

# Same search order as env.py's find_data(): ROOT/data, then ROOT.
DATA_OK=0
for candidate in "$ROOT_DIR/data" "$ROOT_DIR"; do
    if [ -f "$candidate/kernel_payload.json" ] && [ -f "$candidate/recency_results.json" ]; then
        DATA_OK=1
        break
    fi
done
[ "$DATA_OK" -eq 1 ] || fail "required data files (kernel_payload.json, recency_results.json) not found under $ROOT_DIR or $ROOT_DIR/data"

echo "  python3:          $(command -v python3)"
echo "  run_campaign.py:  found at $ENV_DIR/run_campaign.py"
echo "  data files:       found"

echo
echo "== running: python3 run_campaign.py (3 seeds, budget=2000 each -- this takes a bit) =="
echo

cd "$ENV_DIR"
set +e
python3 run_campaign.py
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
    fail "run_campaign.py exited with status $STATUS (see traceback above)"
fi

SUMMARY_JSON="$ENV_DIR/campaign_summary.json"
[ -f "$SUMMARY_JSON" ] || fail "run_campaign.py exited 0 but $SUMMARY_JSON was not written"

echo
echo "== final gap_to_second by seed / judge (from campaign_summary.json) =="
echo

python3 - "$SUMMARY_JSON" <<'PYEOF'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    summaries = json.load(f)

for s in summaries:
    print(f"seed={s['seed']} (stopped: {s.get('stopped_reason')}, "
          f"steps={s.get('steps_taken')}):")
    gaps = s.get("final_gap_by_judge") or {}
    if not gaps:
        print("  (no final_gap_by_judge entries)")
        continue
    for judge in sorted(gaps):
        g = gaps[judge]
        if g is None:
            print(f"  {judge}: n/a (fewer than 2 operators probed)")
        else:
            print(f"  {judge}: leader={g['rank1']} gap={g['gap']}")
PYEOF

echo
echo "PASS: reproduce_core.sh completed -- $SUMMARY_JSON written, gaps summarized above"
exit 0
