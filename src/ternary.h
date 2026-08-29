#ifndef TERNARY_H
#define TERNARY_H

#include <stdint.h>

extern const int8_t TERN_LUT[4];

uint8_t tern_pack4(int8_t a, int8_t b, int8_t c, int8_t d);
int8_t tern_get(const uint8_t *p, int i);
int8_t tern_quant(float w, float tau);
int32_t tern_dot(const int8_t *x, const uint8_t *w, int n);

void tern_set(uint8_t *p, int i, int8_t v);
void tern_adjust(uint8_t *p, int i, int dir);

#endif