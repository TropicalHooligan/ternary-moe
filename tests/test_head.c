#include <stdio.h>
#include "head.h"
#include "linear.h"

static int test_head_bytes(void) {
    return head_weight_bytes(4, 4) == 4 &&
           head_weight_bytes(4, 5) == 8 &&
           head_weight_bytes(256, 32) == 2048;
}

static int test_head_logits(void) {
    uint8_t w[4];
    int8_t x[4] = {1, 2, 3, 4};
    int32_t logits[4];
    int32_t ref[4];

    head_fill_pattern_seed(w, 4, 4, 3);
    head_logits_i32(logits, w, x, 4, 4);
    tern_linear_i32(ref, w, x, 4, 4);

    return logits[0] == ref[0] &&
           logits[1] == ref[1] &&
           logits[2] == ref[2] &&
           logits[3] == ref[3];
}

int main(void) {
    int ok = 1;
    ok &= test_head_bytes();
    ok &= test_head_logits();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}