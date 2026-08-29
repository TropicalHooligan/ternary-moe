#include <stdio.h>
#include "expert_random.h"

static int test_expert_fill_random(void) {
    uint8_t a1[4], a2[4], b1[4], b2[4];
    expert_fill_random(a1, a2, 4, 4, 7);
    expert_fill_random(b1, b2, 4, 4, 7);
    int same = 1;
    for (int i = 0; i < 4; ++i) {
        if (a1[i] != b1[i] || a2[i] != b2[i]) same = 0;
    }
    int nz1 = 0, nz2 = 0;
    for (int i = 0; i < 4; ++i) {
        if (a1[i] != 0) nz1 = 1;
        if (a2[i] != 0) nz2 = 1;
    }
    return same && nz1 && nz2;
}

int main(void) {
    int ok = 1;
    ok &= test_expert_fill_random();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}