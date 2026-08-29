#include <stdio.h>
#include "act.h"

static int test_relu(void) {
    int8_t x[4] = {-5, 0, 7, -128};
    int8_t y[4];

    act_relu_q8(y, x, 4);

    return y[0] == 0 &&
           y[1] == 0 &&
           y[2] == 7 &&
           y[3] == 0;
}

static int test_relu_inplace(void) {
    int8_t x[3] = {-1, 2, -3};

    act_relu_q8(x, x, 3);

    return x[0] == 0 &&
           x[1] == 2 &&
           x[2] == 0;
}

int main(void) {
    int ok = 1;
    ok &= test_relu();
    ok &= test_relu_inplace();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}