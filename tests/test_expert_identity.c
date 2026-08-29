#include <stdio.h>
#include "expert.h"
#include "expert_identity.h"

static int test_expert_identity(void) {
    uint8_t w1[2] = {255, 255};
    uint8_t w2[2] = {255, 255};
    int8_t x[2] = {5, -3};
    int8_t h[2];
    int8_t out[2];
    int32_t scratch[2];
    expert_fill_identity(w1, w2, 2, 2);
    expert_ffn_q8(out, w1, w2, x, 2, 2, 0, 0, h, scratch);
    return out[0] == 5 && out[1] == 0;
}

int main(void) {
    int ok = 1;
    ok &= test_expert_identity();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}