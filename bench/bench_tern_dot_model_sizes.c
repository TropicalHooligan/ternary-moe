/*
 * bench_ternary.c exercises tern_dot at n=1024, which is convenient for
 * throughput numbers but bigger than anything the model actually uses.
 * This one benchmarks it at the real row widths from train_float.c
 * (DIM=32 for the head/context vectors, HID=64 for the expert FFN
 * hidden layer) so the numbers map directly onto training/inference
 * cost instead of a synthetic size.
 */
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "ternary.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void run_one(int n, int reps) {
    int8_t x[128];
    uint8_t w[32];
    volatile int64_t sink = 0;

    for (int i = 0; i < n; ++i) x[i] = (int8_t)(i * 3 - 61);
    for (int b = 0; b < (n + 3) / 4; ++b) w[b] = (uint8_t)(b * 41 + 7);

    double t0 = now_sec();
    for (int r = 0; r < reps; ++r) sink += tern_dot(x, w, n);
    double dt = now_sec() - t0;
    if (dt <= 0.0) dt = 1e-9;

    printf("n=%d reps=%d avg_ns=%.2f dots/s=%.0f\n",
           n, reps, dt * 1e9 / reps, (double)reps / dt);
    (void)sink;
}

int main(void) {
    run_one(32, 2000000);  /* DIM  */
    run_one(64, 2000000);  /* HID  */
    return 0;
}
