#ifndef ACT_H
#define ACT_H

#include <stdint.h>

void act_relu_q8(int8_t *y, const int8_t *x, int n);

#endif