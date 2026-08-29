#include "moe.h"
#include "expert.h"
#include "router.h"
#include "mix.h"

int moe_w1_total_bytes(int experts, int dim, int hidden) {
    return experts * expert_w1_bytes(dim, hidden);
}

int moe_w2_total_bytes(int experts, int dim, int hidden) {
    return experts * expert_w2_bytes(dim, hidden);
}

void moe_fill_pattern(uint8_t *w1_all, uint8_t *w2_all, int experts, int dim, int hidden) {
    int wb1 = expert_w1_bytes(dim, hidden);
    int wb2 = expert_w2_bytes(dim, hidden);
    for (int e = 0; e < experts; ++e) {
        expert_fill_pattern_seed(w1_all + e * wb1,
                                 w2_all + e * wb2,
                                 dim,
                                 hidden,
                                 e * 5 + 1);
    }
}

void moe_forward_top1(int8_t *out,
                      const uint8_t *w1_all,
                      const uint8_t *w2_all,
                      const float *scores,
                      const int8_t *x,
                      int experts,
                      int dim,
                      int hidden,
                      int shift1,
                      int shift2,
                      int8_t *h,
                      int32_t *scratch) {
    if (experts <= 0) {
        for (int i = 0; i < dim; ++i) out[i] = 0;
        return;
    }
    int e = router_top1(scores, experts);
    int wb1 = expert_w1_bytes(dim, hidden);
    int wb2 = expert_w2_bytes(dim, hidden);
    expert_ffn_q8(out, w1_all + e * wb1, w2_all + e * wb2, x, dim, hidden, shift1, shift2, h, scratch);
}

void moe_forward_top2(int8_t *out,
                      int8_t *tmp,
                      const uint8_t *w1_all,
                      const uint8_t *w2_all,
                      const float *scores,
                      const int8_t *x,
                      int experts,
                      int dim,
                      int hidden,
                      int shift1,
                      int shift2,
                      int8_t *h1,
                      int8_t *h2,
                      int32_t *scratch) {
    if (experts <= 0) {
        for (int i = 0; i < dim; ++i) out[i] = 0;
        return;
    }
    int e0 = 0;
    int e1 = 0;
    router_top2(scores, experts, &e0, &e1);
    int wb1 = expert_w1_bytes(dim, hidden);
    int wb2 = expert_w2_bytes(dim, hidden);
    expert_ffn_q8(out, w1_all + e0 * wb1, w2_all + e0 * wb2, x, dim, hidden, shift1, shift2, h1, scratch);
    if (e1 == e0) return;
    expert_ffn_q8(tmp, w1_all + e1 * wb1, w2_all + e1 * wb2, x, dim, hidden, shift1, shift2, h2, scratch);
    mix_avg_q8(out, tmp, dim);
}