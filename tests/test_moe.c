#include <stdio.h>
#include "moe.h"
#include "expert.h"
#include "mix.h"

static int test_moe_bytes(void) {
    return moe_w1_total_bytes(2, 4, 4) == 8 &&
           moe_w2_total_bytes(2, 4, 4) == 8;
}

static int test_moe_top1_routes(void) {
    enum { E = 2, DIM = 4, HID = 4 };
    uint8_t w1[8], w2[8];
    int8_t x[4] = {1, 2, 3, 4};
    int8_t out[4], h[4], h2[4], ref[4];
    int32_t scratch[4];
    float scores[E] = {0.1f, 0.9f};
    int wb1 = expert_w1_bytes(DIM, HID), wb2 = expert_w2_bytes(DIM, HID);

    moe_fill_pattern(w1, w2, E, DIM, HID);
    moe_forward_top1(out, w1, w2, scores, x, E, DIM, HID, 0, 0, h, scratch);
    expert_ffn_q8(ref, w1 + wb1, w2 + wb2, x, DIM, HID, 0, 0, h2, scratch);

    return out[0] == ref[0] &&
           out[1] == ref[1] &&
           out[2] == ref[2] &&
           out[3] == ref[3];
}

static int test_moe_top2_routes(void) {
    enum { E = 2, DIM = 4, HID = 4 };
    uint8_t w1[8], w2[8];
    int8_t x[4] = {1, 2, 3, 4};
    int8_t out[4], tmp[4], ref0[4], ref1[4], h0[4], h1[4], h2[4];
    int32_t scratch[4];
    float scores[E] = {0.9f, 0.8f};
    int wb1 = expert_w1_bytes(DIM, HID), wb2 = expert_w2_bytes(DIM, HID);

    moe_fill_pattern(w1, w2, E, DIM, HID);
    moe_forward_top2(out, tmp, w1, w2, scores, x, E, DIM, HID, 0, 0, h0, h1, scratch);
    expert_ffn_q8(ref0, w1, w2, x, DIM, HID, 0, 0, h2, scratch);
    expert_ffn_q8(ref1, w1 + wb1, w2 + wb2, x, DIM, HID, 0, 0, h2, scratch);
    mix_avg_q8(ref0, ref1, DIM);

    return out[0] == ref0[0] &&
           out[1] == ref0[1] &&
           out[2] == ref0[2] &&
           out[3] == ref0[3];
}

int main(void) {
    int ok = 1;
    ok &= test_moe_bytes();
    ok &= test_moe_top1_routes();
    ok &= test_moe_top2_routes();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}