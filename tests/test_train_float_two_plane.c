#include <stdio.h>
#include "train_float.h"

static int test_train_float_two_plane(void) {
    const char *path = "build/tmp_float_two_plane.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(path) <= 16) return 0;
    train_float_init();
    train_float_epoch();
    train_float_epoch();
    float loss = train_float_eval_two_plane();
    return loss >= 0.0f && loss < 30.0f;
}

int main(void) {
    int ok = 1;
    ok &= test_train_float_two_plane();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}