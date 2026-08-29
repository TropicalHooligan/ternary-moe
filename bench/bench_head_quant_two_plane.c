#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "head_quant.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { VOCAB = 256, DIM = 32, RB = (DIM + 3) / 4, REPS = 30 };
    static float w[VOCAB * DIM];
    static uint8_t p0[VOCAB * RB];
    static uint8_t p1[VOCAB * RB];
    static float s0[VOCAB];
    static float s1[VOCAB];
    for (int i = 0; i < VOCAB * DIM; ++i) w[i] = (i & 1) ? 0.02f : -0.02f;
    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        head_quantize_two_plane(w, p0, p1, s0, s1, VOCAB, DIM);
    }
    double dt = now_sec() - t0;
    if (dt <= 0.0) dt = 1e-9;
    printf("head_quantize_two_plane vocab=%d dim=%d reps=%d avg_ns=%.2f quants/s=%.0f\n", VOCAB, DIM, REPS, dt * 1e9 / REPS, (double)REPS / dt);
    return 0;
}