#!/usr/bin/env bash
# Runs every test binary passed on the command line, collects PASS/FAIL,
# and exits non-zero if anything failed. Each test/*.c prints "PASS" or
# "FAIL" on its own stdout line and returns 0/1 -- this just aggregates.
set -u

pass=0
fail=0
failed_names=()

# Temp files used by a few tests (e.g. checkpoint round-trips) live here.
mkdir -p build

for bin in "$@"; do
    name=$(basename "$bin")
    start=$(date +%s.%N)
    out=$("$bin" 2>&1)
    rc=$?
    end=$(date +%s.%N)
    ms=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.1f", (b-a)*1000}')

    if [ $rc -eq 0 ] && [[ "$out" == *PASS* ]]; then
        printf "  \033[32mPASS\033[0m  %-42s (%s ms)\n" "$name" "$ms"
        pass=$((pass + 1))
    else
        printf "  \033[31mFAIL\033[0m  %-42s (%s ms)\n" "$name" "$ms"
        [ -n "$out" ] && printf "        -> %s\n" "$out"
        fail=$((fail + 1))
        failed_names+=("$name")
    fi
done

total=$((pass + fail))
echo "----------------------------------------------------------------"
echo "  $pass/$total tests passed"

if [ $fail -gt 0 ]; then
    echo "  failed: ${failed_names[*]}"
    exit 1
fi
exit 0
