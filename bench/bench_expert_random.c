#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "expert_random.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { DIM = 32, HID = 64, RB1 = (DIM + 3) / 4, RB2 = (HID + 3) / 4, REPS = 20 };
    static uint8_t w1[HID * RB1];
    static uint8_t w2[DIM * RB2];
    volatile int sink = 0;
    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        expert_fill_random(w1, w2, DIM, HID, r + 1);
        sink += w1[0];
    }
    double dt = now_sec() - t0;
    if (dt <= 0.0) dt = 1e-9;
    printf("expert_fill_random dim=%d hidden=%d reps=%d avg_ns=%.2f fills/s=%.0f\n", DIM, HID, REPS, dt * 1e9 / REPS, (double)REPS / dt);
    (void)sink;
    return 0;
}