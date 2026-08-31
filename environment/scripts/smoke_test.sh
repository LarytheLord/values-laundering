#!/usr/bin/env bash
# Thin wrapper around `python3 env.py --selftest`.
#
# What this adds over calling the Python directly: environment sanity checks
# (python3 on PATH, env.py present, the two frozen data files present) with a
# readable error and non-zero exit instead of a raw Python traceback, and a
# single PASS/FAIL line at the end so a reviewer can tell success from the
# last line of output alone.
#
# No environment logic lives here. All it does is check paths exist, call
# `python3 env.py --selftest` (env.py's own documented smoke test: one seed,
# budget=300, ~84-step trace, "All assertions hold" on success -- see
# environment/README.md), and report PASS/FAIL based on that command's own
# output and exit code.
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

echo "== smoke_test.sh: environment sanity checks =="

command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH"

[ -f "$ENV_DIR/env.py" ] || fail "env.py not found at $ENV_DIR/env.py"

# env.py's find_data() searches ROOT and ROOT/data (repo root above
# environment/) for these two frozen files; mirror that search here so the
# check fails in the same place env.py would, but with a plain-English reason.
DATA_OK=0
for candidate in "$ROOT_DIR/data" "$ROOT_DIR"; do
    if [ -f "$candidate/kernel_payload.json" ] && [ -f "$candidate/recency_results.json" ]; then
        DATA_OK=1
        break
    fi
done
[ "$DATA_OK" -eq 1 ] || fail "required data files (kernel_payload.json, recency_results.json) not found under $ROOT_DIR or $ROOT_DIR/data"

echo "  python3:    $(command -v python3)"
echo "  env.py:     found at $ENV_DIR/env.py"
echo "  data files: found"

echo
echo "== running: python3 env.py --selftest =="
echo

cd "$ENV_DIR"
set +e
OUTPUT="$(python3 env.py --selftest 2>&1)"
STATUS=$?
set -e

echo "$OUTPUT"
echo

if [ "$STATUS" -ne 0 ]; then
    echo "FAIL: env.py --selftest exited with status $STATUS (see traceback above)" >&2
    exit 1
fi

if ! echo "$OUTPUT" | grep -q "All assertions hold"; then
    echo "FAIL: env.py --selftest exited 0 but never printed 'All assertions hold'" >&2
    exit 1
fi

echo "PASS: smoke test succeeded (selftest completed, all assertions hold)"
exit 0
