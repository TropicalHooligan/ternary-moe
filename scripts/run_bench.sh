#!/usr/bin/env bash
# Runs every benchmark binary passed on the command line and prints its
# one-line result. Benches that need scratch files (e.g. bench_train_*)
# write under build/, so make sure that exists.
set -u
mkdir -p build

printf "  %-38s %s\n" "benchmark" "result"
echo "----------------------------------------------------------------------"
for bin in "$@"; do
    name=$(basename "$bin")
    line=$("$bin" 2>&1)
    printf "  %-38s %s\n" "$name" "$line"
done
