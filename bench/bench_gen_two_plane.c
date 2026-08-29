#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "train_float.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const char *path = "build/bench_float_gen.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    for (int i = 0; i < 256; ++i) fputc('a' + (i & 25), f);
    fclose(f);
    train_float_load(path);
    train_float_init();
    train_float_prepare_two_plane();
    uint8_t ctx[16];
    int len = train_float_copy_tail(ctx, 16);
    volatile int sink = 0;
    double t0 = now_sec();
    for (int r = 0; r < 1000; ++r) {
        sink += train_float_next_two_plane(ctx, len);
    }
    double dt = now_sec() - t0;
    if (dt <= 0.0) dt = 1e-9;
    printf("train_float_next_two_plane calls=1000 avg_ns=%.2f calls/s=%.0f\n", dt * 1e9 / 1000.0, 1000.0 / dt);
    (void)sink;
    return 0;
}