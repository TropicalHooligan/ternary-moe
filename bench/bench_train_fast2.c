#include <stdio.h>
#include <time.h>
#include "train_float.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const char *path = "build/bench_fast2.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    for (int i = 0; i < 512; ++i) fputc('a' + (i & 25), f);
    fclose(f);
    train_float_load(path);
    train_float_init();
    train_float_fast_prepare();
    double t0 = now_sec();
    float loss = train_float_qat_two_plane_epoch_fast2();
    double dt = now_sec() - t0;
    if (dt <= 0.0) dt = 1e-9;
    printf("train_float_qat_two_plane_epoch_fast2 avg_ns=%.2f epoch/s=%.2f loss=%.4f\n", dt * 1e9, 1.0 / dt, loss);
    (void)loss;
    return 0;
}