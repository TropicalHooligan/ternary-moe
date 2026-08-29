/*
 * Property test for the optimized tern_dot(): for many random ternary
 * rows and input vectors, tern_dot's result must match a naive
 * per-element reference built directly on tern_get(). This is the
 * correctness net for the branchless/unrolled rewrite in ternary.c --
 * if the fast path and the element-by-element path ever disagree,
 * this is what should catch it.
 */
#include <stdio.h>
#include <stdint.h>
#include "ternary.h"

static uint32_t rng_state = 0xC0FFEEu;

static uint32_t next_rand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static int32_t naive_dot(const int8_t *x, const uint8_t *w, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; ++i) {
        s += (int32_t)x[i] * (int32_t)tern_get(w, i);
    }
    return s;
}

static int check_len(int n, int trials) {
    int rb = (n + 3) / 4;
    int8_t x[256];
    uint8_t w[64];

    for (int t = 0; t < trials; ++t) {
        for (int i = 0; i < n; ++i) x[i] = (int8_t)((next_rand() % 255) - 127);
        for (int b = 0; b < rb; ++b) w[b] = (uint8_t)next_rand();

        int32_t got = tern_dot(x, w, n);
        int32_t want = naive_dot(x, w, n);
        if (got != want) {
            printf("mismatch n=%d trial=%d got=%d want=%d\n", n, t, got, want);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    int ok = 1;
    /* Exact multiples of 4 (the fast unrolled path) ... */
    ok &= check_len(4, 200);
    ok &= check_len(32, 200);
    ok &= check_len(64, 200);
    ok &= check_len(128, 100);
    /* ... and non-multiples, to exercise the scalar tail loop. */
    ok &= check_len(5, 200);
    ok &= check_len(30, 200);
    ok &= check_len(63, 100);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
