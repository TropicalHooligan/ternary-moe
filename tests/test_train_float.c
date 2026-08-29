#include <stdio.h>
#include "train_float.h"

static int test_train_float_learns(void) {
    const char *path = "build/tmp_float.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(path) <= 16) return 0;
    train_float_init();
    float first = train_float_epoch();
    float last = first;
    for (int i = 0; i < 4; ++i) last = train_float_epoch();
    return last < first;
}

int main(void) {
    int ok = 1;
    ok &= test_train_float_learns();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}