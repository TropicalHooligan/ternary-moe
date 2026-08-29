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
    enum { N = 256, REPS = 2000 };
    static uint8_t p[N / 4];
    volatile int64_t sink = 0;

    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        for (int i = 0; i < N; ++i) {
            tern_adjust(p, i, ((r + i) & 1) ? 1 : -1);
        }
        sink += tern_get(p, 0);
    }
    double dt = now_sec() - t0;

    if (dt <= 0.0) dt = 1e-9;

    printf("tern_adjust n=%d reps=%d avg_ns=%.2f adjusts/s=%.0f\n",
           N, REPS, dt * 1e9 / (REPS * (double)N), (double)REPS * N / dt);

    (void)sink;
    return 0;
}