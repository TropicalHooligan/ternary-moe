#ifndef EXPERT_H
#define EXPERT_H

#include <stdint.h>

int expert_row_bytes(int in_dim);
int expert_w1_bytes(int dim, int hidden);
int expert_w2_bytes(int dim, int hidden);

void expert_fill_pattern(uint8_t *w1, uint8_t *w2, int dim, int hidden);
void expert_fill_pattern_seed(uint8_t *w1, uint8_t *w2, int dim, int hidden, int seed);

void expert_ffn_q8(int8_t *out,
                   const uint8_t *w1,
                   const uint8_t *w2,
                   const int8_t *x,
                   int dim,
                   int hidden,
                   int shift1,
                   int shift2,
                   int8_t *h,
                   int32_t *scratch);

#endif