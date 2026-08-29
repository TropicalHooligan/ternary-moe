#include "expert.h"
#include "linear.h"
#include "act.h"

int expert_row_bytes(int in_dim) {
    return (in_dim + 3) / 4;
}

int expert_w1_bytes(int dim, int hidden) {
    return hidden * expert_row_bytes(dim);
}

int expert_w2_bytes(int dim, int hidden) {
    return dim * expert_row_bytes(hidden);
}

void expert_fill_pattern(uint8_t *w1, uint8_t *w2, int dim, int hidden) {
    tern_linear_fill_pattern(w1, hidden, dim);
    tern_linear_fill_pattern(w2, dim, hidden);
}

void expert_fill_pattern_seed(uint8_t *w1, uint8_t *w2, int dim, int hidden, int seed) {
    tern_linear_fill_pattern_seed(w1, hidden, dim, seed);
    tern_linear_fill_pattern_seed(w2, dim, hidden, seed + 1);
}

void expert_ffn_q8(int8_t *out,
                   const uint8_t *w1,
                   const uint8_t *w2,
                   const int8_t *x,
                   int dim,
                   int hidden,
                   int shift1,
                   int shift2,
                   int8_t *h,
                   int32_t *scratch) {
    tern_linear_q8(h, w1, x, hidden, dim, shift1, scratch);
    act_relu_q8(h, h, hidden);
    tern_linear_q8(out, w2, h, dim, hidden, shift2, scratch);
}