#include <stdio.h>
#include "head_quant.h"
#include "ternary.h"

static int test_head_quantize_scaled(void) {
    float w[4] = {0.2f, -0.2f, 0.01f, 0.0f};
    uint8_t p[1] = {0};
    float s[1] = {0.0f};
    head_quantize_scaled(w, p, s, 1, 4);
    return tern_get(p, 0) == 1 &&
           tern_get(p, 1) == -1 &&
           tern_get(p, 2) == 0 &&
           s[0] > 0.19f && s[0] < 0.21f;
}

int main(void) {
    int ok = 1;
    ok &= test_head_quantize_scaled();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}