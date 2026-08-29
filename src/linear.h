#ifndef LINEAR_H
#define LINEAR_H

#include <stdint.h>

void tern_linear_fill_pattern(uint8_t *w, int out_dim, int in_dim);
void tern_linear_fill_pattern_seed(uint8_t *w, int out_dim, int in_dim, int seed);
void tern_linear_i32(int32_t *y, const uint8_t *w, const int8_t *x, int out_dim, int in_dim);
void linear_i32_to_q8(int8_t *y, const int32_t *acc, int n, int shift);
void tern_linear_q8(int8_t *y, const uint8_t *w, const int8_t *x, int out_dim, int in_dim, int shift, int32_t *scratch);

#endif