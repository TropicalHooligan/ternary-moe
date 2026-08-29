#include <stdio.h>
#include "mix.h"

static int test_mix_avg(void) {
    int8_t a[4] = {1, 2, -2, -128};
    int8_t b[4] = {3, -2, 2, -128};

    mix_avg_q8(a, b, 4);

    return a[0] == 2 &&
           a[1] == 0 &&
           a[2] == 0 &&
           a[3] == -128;
}

int main(void) {
    int ok = 1;
    ok &= test_mix_avg();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}