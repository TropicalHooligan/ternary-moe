#include <stdio.h>
#include "ternary.h"

static int test_tern_set_get(void) {
    uint8_t p[1] = {0};

    tern_set(p, 0, 1);
    tern_set(p, 1, -1);
    tern_set(p, 2, 0);
    tern_set(p, 3, 1);

    return tern_get(p, 0) == 1 &&
           tern_get(p, 1) == -1 &&
           tern_get(p, 2) == 0 &&
           tern_get(p, 3) == 1;
}

static int test_tern_adjust(void) {
    uint8_t p[1] = {0};

    tern_adjust(p, 0, 1);
    tern_adjust(p, 0, 1);
    tern_adjust(p, 1, -1);
    tern_adjust(p, 1, -1);

    return tern_get(p, 0) == 1 &&
           tern_get(p, 1) == -1;
}

int main(void) {
    int ok = 1;
    ok &= test_tern_set_get();
    ok &= test_tern_adjust();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}