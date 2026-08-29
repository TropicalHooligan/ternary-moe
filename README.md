# ternary-moe

A small byte-level language model in plain C: ternary (2-bit, {-1, 0, +1})
weights, a tiny mixture-of-experts feed-forward block, and a
quantization-aware-trained output head. No dependencies beyond libc + libm.
Runs happily on a phone (this started life as a Termux project).

This is the cleaned-up, benchmarked, and unit-tested version of the
original dump. Nothing about the model or the training algorithm changed —
same architecture, same math, same checkpoint format — but the hot path
runs measurably faster and the project now builds and tests itself with a
single command.

## What's actually in here

- **`src/`** — the library. Ternary bit-packing (`ternary.c`), a linear
  layer built on it (`linear.c`), a 2-layer ternary FFN "expert"
  (`expert.c`), a 4-expert top-1/top-2 MoE router (`moe.c`, `router.c`),
  a float32 output head that gets quantized to two ternary "planes" for
  inference (`head_f32.c`, `head_quant.c`), softmax/cross-entropy
  (`loss.c`), and the training/generation orchestration for all of the
  above (`train_float.c`).
- **`app/`** — 10 small CLI programs: train from scratch, continue from a
  checkpoint, train with random or trainable experts, generate text from
  a saved model.
- **`tests/`** — 31 unit tests, one binary per test file, each printing
  `PASS`/`FAIL`.
- **`bench/`** — 13 microbenchmarks measuring the actual hot loops.
- **`data/`** — the original dump didn't include any data files, only
  their names in the project listing. `repeat.txt` here is a small
  placeholder. `sample.txt` isn't shipped at all: `./run.sh demo`
  downloads a real one automatically the first time you run it (see
  below), so you never train on synthetic word-salad by default.

## Build & run

```sh
make            # library + all 10 apps -> bin/
make test       # build + run all 31 tests, print a summary
make bench      # build + run all 13 benchmarks, print a summary
make all        # apps + test + bench
make clean
```

Or the one-click version, which also runs a short end-to-end training +
generation demo on `data/sample.txt`:

```sh
./run.sh              # build, test, bench, demo -- everything
./run.sh build         # just build
./run.sh test          # build + test
./run.sh bench         # build + bench
./run.sh demo          # build + a short train/generate demo
./run.sh clean
```

`make` probes for `-march=native` support before using it (some
cross/Termux toolchains reject it), so the build shouldn't hard-fail on
an unusual compiler. Override manually if needed:

```sh
make CFLAGS_ARCH=            # disable native arch tuning
make CC=clang                # use a specific compiler
```

`run.sh` always runs `make` for you first — every subcommand except
`clean` rebuilds before doing anything else, so the binaries you test
or benchmark always match what's on disk. It also shows a spinner and
a timing line for each step instead of a wall of raw build output:

```
==> Building library + apps
  ✔  make apps                                       0.4s

==> Running test suite
  ✔  test_act                                   (2.2 ms)
  ...
  31/31 tests passed
```

**Training data for the demo:** `./run.sh demo` (and `./run.sh all`)
need something to train on. If `data/sample.txt` doesn't exist yet, the
script downloads a small public-domain corpus automatically — first
try is [tiny-shakespeare](https://github.com/karpathy/char-rnn), the
standard toy dataset for exactly this kind of byte/char-level model
demo; if that host is unreachable it falls back to a Project Gutenberg
text. Only the model's fixed 16 KB input buffer worth of it is ever
used. If there's no network at all, `run.sh` generates a small
synthetic placeholder locally instead (pure `awk`, no Python needed)
and tells you it did so — replace it with real text once you're online.

## Using the apps

```sh
# train a single-expert model from scratch, save a checkpoint
bin/train_save_checkpoint data/sample.txt build/model.tmo [normal_epochs] [qat_epochs]

# train a 4-expert MoE model (fixed random experts) from scratch
bin/train_save_moe data/sample.txt build/model.tmo [normal_epochs] [qat_epochs]

# continue training an existing checkpoint on new data
bin/train_continue_checkpoint build/model.tmo data/more.txt [normal] [qat]
bin/train_continue_moe        build/model.tmo data/more.txt [normal] [qat]

# train with the experts themselves updated via a ternary bit-flip rule
bin/train_moe_expert0  data/sample.txt build/model.tmo [normal] [qat] [threshold]
bin/train_moe_experts  data/sample.txt build/model.tmo [normal] [qat] [threshold]

# generate text from a checkpoint, seeded with a file's tail bytes
bin/gen_from_checkpoint build/model.tmo seed.txt [gen_len]
bin/gen_moe_checkpoint  build/model.tmo seed.txt [gen_len]
bin/gen_moe_expert0     build/model.tmo seed.txt [gen_len]
```

## Architecture, briefly

Each byte position gets embedded and averaged over the previous 16 bytes
into a 32-dim int8 context vector. That vector is routed to one (or a
soft-mixed top-2) of 4 experts, each a ternary-weight 32→64→32
feed-forward block with ReLU. The expert's output feeds a dense float32
output head (256×32) that produces logits over the byte vocabulary. The
head is periodically re-quantized into two ternary "planes" (a coarse
pass plus a residual correction pass) so inference can run on pure
ternary dot products while training keeps the precision of float32
gradients — a small from-scratch take on quantization-aware training.

## What changed from the original dump

**Correctness first:** every change here was validated against the
original 30 unit tests before and after, plus one new property test
(below) — nothing here changes model behavior, checkpoint format, or CLI
argument order.

1. **`tern_dot` is ~4.8x faster.** This is the single hottest function
   in the codebase — every ternary linear layer, every MoE expert
   forward pass, and every quantized-head logit computation for all 256
   vocab entries goes through it. The original decoded each 2-bit
   ternary weight with a table lookup (`TERN_LUT[(w[i>>2] >> ...) & 3]`)
   recomputed per element. The rewrite:
   - decodes a code with branchless arithmetic (`(code&1) - ((code>>1)&1)`)
     instead of a memory-indirect table lookup, and
   - walks whole packed bytes, unrolling all 4 lanes per byte instead of
     recomputing `i>>2` / `i&3` on every single element.

   Measured on this container (`bench/bench_ternary.c`, n=1024,
   10,000 reps): **885 ns/call → 183 ns/call**. On a more realistic
   generation workload (`bench/bench_gen_two_plane.c`, which calls
   `tern_dot` 512 times per generated byte via the two-plane head):
   **48.7k calls/s → 68.8k calls/s** (+41%). Exact numbers will differ
   on Termux/ARM; re-run `make bench` there to get real figures for
   your device — the microbenchmark in `bench_tern_dot_model_sizes.c`
   uses the model's actual DIM=32/HID=64 widths instead of a synthetic
   size.

2. **Deduplicated repeated code.**
   - `append_ctx()` was copy-pasted verbatim into 4 different `app/*.c`
     files; it's now `app/common/gen_ctx.h`, included where needed.
   - The "score a context vector against the two-plane quantized head"
     loop was inlined 4 separate times across `train_float.c` (once as
     a genuinely unreadable single line). It's now the one
     `two_plane_logits_i32()` helper everywhere.

3. **Readability.** A few lines had grown to 200+ characters cramming
   4-5 statements together (`if (normal <= 0) normal = 1; if (qat <= 0)
   ...`, and worse in the two-plane argmax loops). Reformatted to one
   statement per line; behavior is unchanged.

4. **Build system.** There was no working build script in the dump
   (only `build.sh`/`run.sh` filenames, no content) — this adds a real
   `Makefile` (parallel-buildable, `make -jN` works), `run.sh`, and
   `scripts/run_tests.sh` / `scripts/run_bench.sh` so you don't run 44
   binaries by hand.

5. **`run.sh` always builds, shows real progress, and fetches its own
   demo data.** Every subcommand (except `clean`) runs `make` before
   doing anything else — no silent "using whatever was built last
   time." Long steps (`make`, dataset download) animate a spinner and
   finish with a ✔/✘ and elapsed time instead of dumping raw compiler
   output. And `./run.sh demo` no longer needs a bundled sample file:
   it fetches a small public-domain corpus on first run and only falls
   back to a local synthetic one if there's no network at all.

6. **New tests & benchmarks.**
   - `tests/test_ternary_dot_property.c` — checks the optimized
     `tern_dot` against a naive per-element reference across random
     inputs, at both multiple-of-4 lengths (the fast unrolled path) and
     non-multiples (the scalar tail path). This is the test that would
     catch a regression in the optimization above.
   - `bench/bench_tern_dot_model_sizes.c` — benchmarks `tern_dot` at the
     model's real DIM=32 / HID=64 widths, since the existing
     `bench_ternary.c` uses a synthetic n=1024.

## Notes / things worth knowing

- All state in `train_float.c` is static/global (single model instance
  per process) — that's inherited from the original design, not
  something this pass changed. Fine for a CLI tool, not thread-safe.
- `DATA_MAX` is 16384 bytes (see the `enum` in `train_float.c`) —
  training data larger than that gets silently truncated by
  `data_load_file`. Worth raising if you point it at a bigger corpus.
- The placeholder `data/sample.txt` is synthetic word-salad, not real
  text — good enough to prove the pipeline works end-to-end (see
  `./run.sh demo`), not to judge generation quality.
