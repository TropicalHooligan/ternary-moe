#include "linear.h"
#include "ternary.h"

static int8_t pattern_val(int o, int i, int in_dim) {
    if (i >= in_dim) return 0;
    return ((o + i) & 1) ? 1 : -1;
}

static int8_t pattern_val_seed(int seed, int o, int i, int in_dim) {
    if (i >= in_dim) return 0;
    int v = (seed * 7 + o * 3 + i) & 3;
    if (v == 0) return 0;
    if (v == 1) return 1;
    return -1;
}

void tern_linear_fill_pattern(uint8_t *w, int out_dim, int in_dim) {
    int rb = (in_dim + 3) / 4;
    for (int o = 0; o < out_dim; ++o) {
        uint8_t *row = w + o * rb;
        for (int b = 0; b < rb; ++b) {
            int i = b * 4;
            row[b] = tern_pack4(pattern_val(o, i, in_dim),
                                pattern_val(o, i + 1, in_dim),
                                pattern_val(o, i + 2, in_dim),
                                pattern_val(o, i + 3, in_dim));
        }
    }
}

void tern_linear_fill_pattern_seed(uint8_t *w, int out_dim, int in_dim, int seed) {
    int rb = (in_dim + 3) / 4;
    for (int o = 0; o < out_dim; ++o) {
        uint8_t *row = w + o * rb;
        for (int b = 0; b < rb; ++b) {
            int i = b * 4;
            row[b] = tern_pack4(pattern_val_seed(seed, o, i, in_dim),
                                pattern_val_seed(seed, o, i + 1, in_dim),
                                pattern_val_seed(seed, o, i + 2, in_dim),
                                pattern_val_seed(seed, o, i + 3, in_dim));
        }
    }
}

void tern_linear_i32(int32_t *y, const uint8_t *w, const int8_t *x, int out_dim, int in_dim) {
    int rb = (in_dim + 3) / 4;
    for (int o = 0; o < out_dim; ++o) {
        y[o] = tern_dot(x, w + o * rb, in_dim);
    }
}

void linear_i32_to_q8(int8_t *y, const int32_t *acc, int n, int shift) {
    for (int i = 0; i < n; ++i) {
        int32_t v = acc[i] >> shift;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        y[i] = (int8_t)v;
    }
}

void tern_linear_q8(int8_t *y, const uint8_t *w, const int8_t *x, int out_dim, int in_dim, int shift, int32_t *scratch) {
    tern_linear_i32(scratch, w, x, out_dim, in_dim);
    linear_i32_to_q8(y, scratch, out_dim, shift);
}