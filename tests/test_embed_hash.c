#include <stdio.h>
#include "embed.h"

static int test_embed_fill_hash(void) {
    int8_t a[32];
    int8_t b[32];
    embed_fill_hash(a, 4, 8, 64);
    embed_fill_hash(b, 4, 8, 64);
    for (int i = 0; i < 32; ++i) {
        if (a[i] != b[i]) return 0;
        if (a[i] != 64 && a[i] != -64) return 0;
    }
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= test_embed_fill_hash();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}