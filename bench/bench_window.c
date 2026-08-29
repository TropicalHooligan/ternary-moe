#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "window.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { CAP = 16, REPS = 2000000 };
    static uint8_t buf[CAP];
    ByteWindow w;
    window_reset(&w, buf, CAP);

    volatile int64_t sink = 0;

    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        window_push(&w, (uint8_t)r);
        sink += window_get(&w, 0);
    }
    double dt = now_sec() - t0;

    if (dt <= 0.0) dt = 1e-9;

    printf("window_push_get cap=%d reps=%d avg_ns=%.2f ops/s=%.0f\n",
           CAP, REPS, dt * 1e9 / REPS, (double)REPS / dt);

    (void)sink;
    return 0;
}