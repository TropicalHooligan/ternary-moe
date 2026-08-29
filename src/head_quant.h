#ifndef HEAD_QUANT_H
#define HEAD_QUANT_H

#include <stdint.h>

void head_quantize_f32_to_ternary(uint8_t *out, const float *w, int vocab, int dim, float tau);
void head_quantize_scaled(const float *w, uint8_t *pack, float *scales, int vocab, int dim);
void head_quantize_two_plane(const float *w, uint8_t *p0, uint8_t *p1, float *s0, float *s1, int vocab, int dim);

#endif