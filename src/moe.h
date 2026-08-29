#ifndef MOE_H
#define MOE_H

#include <stdint.h>

int moe_w1_total_bytes(int experts, int dim, int hidden);
int moe_w2_total_bytes(int experts, int dim, int hidden);

void moe_fill_pattern(uint8_t *w1_all, uint8_t *w2_all, int experts, int dim, int hidden);

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
                      int32_t *scratch);

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
                      int32_t *scratch);

#endif