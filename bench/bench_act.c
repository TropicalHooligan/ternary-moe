#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "act.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { N = 64, REPS = 5000000 };
    static int8_t x[N];
    static int8_t y[N];
    volatile int64_t sink = 0;

    for (int i = 0; i < N; ++i) x[i] = (int8_t)(i - 32);

    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        act_relu_q8(y, x, N);
        sink += y[0];
    }
    double dt = now_sec() - t0;

    if (dt <= 0.0) dt = 1e-9;

    printf("act_relu_q8 n=%d reps=%d avg_ns=%.2f ops/s=%.0f\n",
           N, REPS, dt * 1e9 / REPS, (double)REPS / dt);

    (void)sink;
    return 0;
}