#include <stdio.h>
#include "train_float.h"

static int test_train_float_gen(void) {
    const char *path = "build/tmp_float_gen.txt";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(path) <= 16) return 0;
    train_float_init();
    train_float_epoch();
    train_float_prepare_two_plane();
    uint8_t ctx[16];
    int len = train_float_copy_tail(ctx, 16);
    int b = train_float_next_two_plane(ctx, len);
    return len > 0 && b >= 0 && b < 256;
}

int main(void) {
    int ok = 1;
    ok &= test_train_float_gen();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}