#include <stdio.h>
#include "head_quant.h"
#include "ternary.h"

static int test_head_quantize_two_plane(void) {
    float w[4] = {0.31f, -0.31f, 0.1f, -0.1f};
    uint8_t p0[1] = {0};
    uint8_t p1[1] = {0};
    float s0[1] = {0.0f};
    float s1[1] = {0.0f};
    head_quantize_two_plane(w, p0, p1, s0, s1, 1, 4);
    return tern_get(p0, 0) == 1 &&
           tern_get(p0, 1) == -1 &&
           tern_get(p0, 2) == 0 &&
           tern_get(p1, 2) == 1 &&
           tern_get(p1, 3) == -1;
}

int main(void) {
    int ok = 1;
    ok &= test_head_quantize_two_plane();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}