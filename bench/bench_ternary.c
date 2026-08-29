#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "ternary.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { N = 1024, REPS = 10000 };
    static int8_t x[N];
    static uint8_t w[N / 4];
    volatile int64_t sink = 0;

    for (int i = 0; i < N; ++i) x[i] = (int8_t)(i & 127);
    for (int i = 0; i < N / 4; ++i) w[i] = (uint8_t)i;

    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        sink += tern_dot(x, w, N);
    }
    double dt = now_sec() - t0;

    if (dt <= 0.0) dt = 1e-9;

    printf("n=%d reps=%d avg_ns=%.1f elems/s=%.0f\n",
           N, REPS, dt * 1e9 / REPS, (double)N * REPS / dt);

    (void)sink;
    return 0;
}