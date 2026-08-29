#include <stdio.h>
#include "train_float.h"

static int test_expert0_train(void) {
    const char *path = "build/tmp_expert0.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(path) <= 16) return 0;
    train_float_init();
    train_float_zero_expert_acc();
    train_float_fast_prepare();
    float first = train_float_epoch_fast_expert0_trainable(8);
    float last = first;
    for (int i = 0; i < 2; ++i) last = train_float_epoch_fast_expert0_trainable(8);
    float qloss = train_float_qat_two_plane_epoch_fast_expert0();
    float eloss = train_float_eval_two_plane_fast_expert0();
    return first >= 0.0f && last >= 0.0f && qloss >= 0.0f && eloss >= 0.0f && eloss < 30.0f;
}

int main(void) {
    int ok = 1;
    ok &= test_expert0_train();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}