#include <stdio.h>
#include "train_float.h"

static int test_train_float_qat_two_plane(void) {
    const char *path = "build/tmp_qat_two_plane.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(path) <= 16) return 0;
    train_float_init();
    train_float_epoch();
    float qloss = train_float_qat_two_plane_epoch();
    float eloss = train_float_eval_two_plane();
    return qloss >= 0.0f && eloss >= 0.0f && eloss < 30.0f;
}

int main(void) {
    int ok = 1;
    ok &= test_train_float_qat_two_plane();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}