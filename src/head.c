#include "head.h"
#include "linear.h"

int head_weight_bytes(int vocab, int dim) {
    return vocab * ((dim + 3) / 4);
}

void head_fill_pattern_seed(uint8_t *w, int vocab, int dim, int seed) {
    tern_linear_fill_pattern_seed(w, vocab, dim, seed);
}

void head_logits_i32(int32_t *logits, const uint8_t *w, const int8_t *x, int vocab, int dim) {
    tern_linear_i32(logits, w, x, vocab, dim);
}