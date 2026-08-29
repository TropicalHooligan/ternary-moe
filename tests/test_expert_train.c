#include <stdio.h>
#include "expert_train.h"
#include "ternary.h"

static int test_expert_update_w2(void) {
    uint8_t w2[2] = {0, 0};
    int16_t acc[4] = {0, 0, 0, 0};
    float grad_y[2] = {1.0f, -1.0f};
    int8_t h[2] = {1, 1};
    expert_update_w2(w2, acc, grad_y, h, 2, 2, 2);
    if (tern_get(w2, 0) != 0) return 0;
    if (tern_get(w2 + 1, 0) != 0) return 0;
    expert_update_w2(w2, acc, grad_y, h, 2, 2, 2);
    return tern_get(w2, 0) == -1 &&
           tern_get(w2, 1) == -1 &&
           tern_get(w2 + 1, 0) == 1 &&
           tern_get(w2 + 1, 1) == 1;
}

static int test_expert_update_w1(void) {
    uint8_t w1[2] = {0, 0};
    uint8_t w2[2] = {0, 0};
    int16_t acc[4] = {0, 0, 0, 0};
    tern_set(w2, 0, 1);
    tern_set(w2 + 1, 1, 1);
    float grad_y[2] = {1.0f, -1.0f};
    int8_t h[2] = {1, 1};
    int8_t x[2] = {1, -1};
    expert_update_w1(w1, acc, w2, grad_y, h, x, 2, 2, 2);
    if (tern_get(w1, 0) != 0) return 0;
    if (tern_get(w1, 1) != 0) return 0;
    expert_update_w1(w1, acc, w2, grad_y, h, x, 2, 2, 2);
    return tern_get(w1, 0) == -1 &&
           tern_get(w1, 1) == 1 &&
           tern_get(w1 + 1, 0) == 1 &&
           tern_get(w1 + 1, 1) == -1;
}

int main(void) {
    int ok = 1;
    ok &= test_expert_update_w2();
    ok &= test_expert_update_w1();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}