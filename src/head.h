#ifndef HEAD_H
#define HEAD_H

#include <stdint.h>

int head_weight_bytes(int vocab, int dim);
void head_fill_pattern_seed(uint8_t *w, int vocab, int dim, int seed);
void head_logits_i32(int32_t *logits, const uint8_t *w, const int8_t *x, int vocab, int dim);

#endif