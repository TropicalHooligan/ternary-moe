#include <stdio.h>
#include "expert.h"

static int test_expert_bytes(void) {
    return expert_row_bytes(4) == 1 &&
           expert_row_bytes(5) == 2 &&
           expert_w1_bytes(4, 4) == 4 &&
           expert_w2_bytes(4, 4) == 4;
}

static int test_expert_ffn(void) {
    uint8_t w1[4];
    uint8_t w2[4];
    int8_t x[4] = {1, 2, 3, 4};
    int8_t h[4];
    int8_t out[4];
    int32_t scratch[4];

    expert_fill_pattern(w1, w2, 4, 4);
    expert_ffn_q8(out, w1, w2, x, 4, 4, 0, 0, h, scratch);

    return out[0] == -4 &&
           out[1] == 4 &&
           out[2] == -4 &&
           out[3] == 4;
}

static int test_expert_ffn_shift(void) {
    uint8_t w1[4];
    uint8_t w2[4];
    int8_t x[4] = {1, 2, 3, 4};
    int8_t h[4];
    int8_t out[4];
    int32_t scratch[4];

    expert_fill_pattern(w1, w2, 4, 4);
    expert_ffn_q8(out, w1, w2, x, 4, 4, 1, 0, h, scratch);

    return out[0] == -2 &&
           out[1] == 2 &&
           out[2] == -2 &&
           out[3] == 2;
}

int main(void) {
    int ok = 1;
    ok &= test_expert_bytes();
    ok &= test_expert_ffn();
    ok &= test_expert_ffn_shift();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}