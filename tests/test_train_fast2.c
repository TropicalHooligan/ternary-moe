#include <stdio.h>
#include "train_float.h"

static int test_train_fast2_qat(void) {
    const char *path = "build/tmp_fast2.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(path) <= 16) return 0;
    train_float_init();
    train_float_fast_prepare();
    train_float_epoch_fast();
    float qloss = train_float_qat_two_plane_epoch_fast2();
    float eloss = train_float_eval_two_plane_fast();
    return qloss >= 0.0f && eloss >= 0.0f && eloss < 30.0f;
}

int main(void) {
    int ok = 1;
    ok &= test_train_fast2_qat();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}