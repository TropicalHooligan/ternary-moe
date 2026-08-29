#!/usr/bin/env bash
######################################################################
# run.sh -- one-click build, test, benchmark and demo for ternary-moe
#
# Usage:
#   ./run.sh              build, test, benchmark, then a training demo
#   ./run.sh build         just build the library + all apps
#   ./run.sh test          build + run the test suite
#   ./run.sh bench         build + run the benchmark suite
#   ./run.sh demo          build + run the training demo only
#   ./run.sh clean         remove build/ and bin/
#
# Every path except `clean` always runs `make` first -- there is no
# "already built, skip it" branch here, so you always know the
# binaries you're about to test/bench/demo match the source on disk.
######################################################################
set -u
cd "$(dirname "$0")"

######################################################################
# Pretty output
######################################################################
if [ -t 1 ]; then
    C_CYAN='\033[1;36m'; C_GREEN='\033[1;32m'; C_RED='\033[1;31m'
    C_DIM='\033[2m';     C_BOLD='\033[1m';     C_RESET='\033[0m'
else
    C_CYAN=''; C_GREEN=''; C_RED=''; C_DIM=''; C_BOLD=''; C_RESET=''
fi

heading() { printf "\n${C_CYAN}==> %s${C_RESET}\n" "$1"; }

# Runs "$@" in the background, animates a spinner on the same line
# while it works, then replaces the spinner with a check/cross and the
# elapsed time. Full output is captured and only shown on failure.
spin_run() {
    local label="$1"; shift
    local log; log="$(mktemp)"
    local frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local start end elapsed i=0

    start=$(date +%s.%N)
    ( "$@" >"$log" 2>&1 ) &
    local pid=$!

    while kill -0 "$pid" 2>/dev/null; do
        i=$(( (i + 1) % ${#frames} ))
        printf "\r  ${C_CYAN}%s${C_RESET}  %s" "${frames:$i:1}" "$label"
        sleep 0.08
    done
    wait "$pid"
    local rc=$?
    end=$(date +%s.%N)
    elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.1fs", b-a}')

    if [ $rc -eq 0 ]; then
        printf "\r  ${C_GREEN}✔${C_RESET}  %-46s ${C_DIM}%s${C_RESET}\n" "$label" "$elapsed"
    else
        printf "\r  ${C_RED}✘${C_RESET}  %-46s ${C_DIM}%s${C_RESET}\n" "$label" "$elapsed"
        echo "----------------------------------------------------------------"
        cat "$log"
        echo "----------------------------------------------------------------"
    fi
    rm -f "$log"
    return $rc
}

######################################################################
# Training data: download a small, real, public-domain corpus if none
# is present yet. tiny-shakespeare is the standard toy dataset for
# exactly this kind of byte/char-level language model demo; a Project
# Gutenberg text is the fallback if that host is unreachable. Only the
# first DATA_MAX (16384) bytes of whatever we get are ever used, since
# that's train_float.c's fixed buffer size, so there's no need to fetch
# more than a short excerpt.
######################################################################
DATA_DIR="data"
SAMPLE_FILE="$DATA_DIR/sample.txt"
MIN_USEFUL_BYTES=256

have_cmd() { command -v "$1" >/dev/null 2>&1; }

fetch_url() {
    local url="$1" out="$2"
    if have_cmd curl; then
        curl -fsSL --max-time 15 "$url" -o "$out" 2>/dev/null
    elif have_cmd wget; then
        wget -q --timeout=15 -O "$out" "$url" 2>/dev/null
    else
        return 1
    fi
}

# Small, synthetic fallback corpus -- used only if we're offline and no
# sample data exists at all, so the demo still has something to chew
# on. Pure bash/awk, no python dependency, so it works on a bare
# Termux install too.
generate_placeholder_corpus() {
    local out="$1"
    local words="the ternary model learns from bytes and routes through experts quickly training a tiny moe network on plain text data weights are packed two bits each into a single byte"
    awk -v words="$words" -v seed="$RANDOM" '
        BEGIN {
            srand(seed);
            n = split(words, w, " ");
            for (line = 0; line < 400; line++) {
                len = 6 + int(rand() * 8);
                out = "";
                for (i = 0; i < len; i++) {
                    out = out (i ? " " : "") w[1 + int(rand() * n)];
                }
                print out;
            }
        }
    ' > "$out"
}

ensure_training_data() {
    if [ -s "$SAMPLE_FILE" ] && [ "$(wc -c < "$SAMPLE_FILE")" -ge "$MIN_USEFUL_BYTES" ]; then
        return 0
    fi

    mkdir -p "$DATA_DIR"
    heading "No training data found -- fetching a small public-domain corpus"

    local tmp; tmp="$(mktemp)"
    local sources=(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        "https://www.gutenberg.org/files/11/11-0.txt"
    )
    local got=0
    for url in "${sources[@]}"; do
        if spin_run "downloading $(basename "$url")" fetch_url "$url" "$tmp"; then
            if [ -s "$tmp" ] && [ "$(wc -c < "$tmp")" -ge "$MIN_USEFUL_BYTES" ]; then
                got=1
                break
            fi
        fi
    done

    if [ "$got" -eq 1 ]; then
        head -c 16384 "$tmp" > "$SAMPLE_FILE"
        echo "  saved $(wc -c < "$SAMPLE_FILE") bytes to $SAMPLE_FILE"
    else
        echo "  no network reachable -- generating a small synthetic placeholder instead"
        generate_placeholder_corpus "$SAMPLE_FILE"
        echo -e "  ${C_DIM}(replace $SAMPLE_FILE with real text once you're online)${C_RESET}"
    fi
    rm -f "$tmp"
}

######################################################################
# Build / test / bench / demo
######################################################################
do_build() {
    heading "Building library + apps"
    spin_run "make apps" make apps
}

do_test() {
    do_build || return 1
    heading "Running test suite"
    make test
}

do_bench() {
    do_build || return 1
    heading "Running benchmark suite"
    make bench
}

do_demo() {
    do_build || return 1
    ensure_training_data
    heading "Demo: training a tiny byte-level MoE model on $SAMPLE_FILE"
    ./bin/train_save_moe "$SAMPLE_FILE" build/demo.tmo 30 30
    heading "Demo: generating 120 bytes from the trained checkpoint"
    printf "the ternary" > build/demo_seed.txt
    ./bin/gen_moe_checkpoint build/demo.tmo build/demo_seed.txt 120
}

cmd="${1:-all}"
case "$cmd" in
    build) do_build ;;
    test)  do_test ;;
    bench) do_bench ;;
    demo)  do_demo ;;
    clean) make clean ;;
    all)
        do_test  && \
        do_bench && \
        do_demo  && \
        heading "Done. Binaries are in bin/, artifacts in build/."
        ;;
    *)
        echo "unknown command: $cmd (expected build|test|bench|demo|clean|all)"
        exit 1
        ;;
esac
