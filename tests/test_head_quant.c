#include <stdio.h>
#include "head_quant.h"
#include "ternary.h"

static int test_head_quantize(void) {
    float w[4] = {0.2f, -0.2f, 0.01f, 0.0f};
    uint8_t out[1] = {0};
    head_quantize_f32_to_ternary(out, w, 1, 4, 0.1f);
    return tern_get(out, 0) == 1 &&
           tern_get(out, 1) == -1 &&
           tern_get(out, 2) == 0 &&
           tern_get(out, 3) == 0;
}

int main(void) {
    int ok = 1;
    ok &= test_head_quantize();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}