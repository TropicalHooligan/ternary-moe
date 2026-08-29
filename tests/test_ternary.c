#include <stdio.h>
#include "ternary.h"

static int test_pack_get(void) {
    uint8_t p = tern_pack4(1, -1, 0, 1);
    if (tern_get(&p, 0) != 1) return 0;
    if (tern_get(&p, 1) != -1) return 0;
    if (tern_get(&p, 2) != 0) return 0;
    if (tern_get(&p, 3) != 1) return 0;
    return 1;
}

static int test_dot(void) {
    int8_t x[4] = {1, 2, 3, -4};
    uint8_t w = tern_pack4(1, -1, 0, 1);
    return tern_dot(x, &w, 4) == -5;
}

int main(void) {
    int ok = 1;
    ok &= test_pack_get();
    ok &= test_dot();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}