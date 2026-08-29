#include <stdio.h>
#include "linear.h"

static int test_linear_i32(void) {
    uint8_t w[2];
    int8_t x[4] = {1, 2, 3, 4};
    int32_t y[2];

    tern_linear_fill_pattern(w, 2, 4);
    tern_linear_i32(y, w, x, 2, 4);

    return y[0] == 2 && y[1] == -2;
}

static int test_linear_q8(void) {
    uint8_t w[2];
    int8_t x[4] = {1, 2, 3, 4};
    int8_t y[2];
    int32_t scratch[2];

    tern_linear_fill_pattern(w, 2, 4);
    tern_linear_q8(y, w, x, 2, 4, 0, scratch);

    return y[0] == 2 && y[1] == -2;
}

static int test_linear_q8_shift(void) {
    uint8_t w[2];
    int8_t x[4] = {1, 2, 3, 4};
    int8_t y[2];
    int32_t scratch[2];

    tern_linear_fill_pattern(w, 2, 4);
    tern_linear_q8(y, w, x, 2, 4, 1, scratch);

    return y[0] == 1 && y[1] == -1;
}

static int test_to_q8_clamp(void) {
    int32_t acc[2] = {300, -300};
    int8_t y[2];

    linear_i32_to_q8(y, acc, 2, 0);

    return y[0] == 127 && y[1] == -127;
}

static int test_to_q8_shift(void) {
    int32_t acc[2] = {2, -2};
    int8_t y[2];

    linear_i32_to_q8(y, acc, 2, 1);

    return y[0] == 1 && y[1] == -1;
}

int main(void) {
    int ok = 1;
    ok &= test_linear_i32();
    ok &= test_linear_q8();
    ok &= test_linear_q8_shift();
    ok &= test_to_q8_clamp();
    ok &= test_to_q8_shift();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}