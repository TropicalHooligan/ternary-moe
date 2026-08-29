#ifndef EXPERT_TRAIN_H
#define EXPERT_TRAIN_H

#include <stdint.h>

void expert_update_w2(uint8_t *w2, int16_t *acc2, const float *grad_y, const int8_t *h, int dim, int hidden, int threshold);
void expert_update_w1(uint8_t *w1, int16_t *acc1, const uint8_t *w2, const float *grad_y, const int8_t *h, const int8_t *x, int dim, int hidden, int threshold);

#endif