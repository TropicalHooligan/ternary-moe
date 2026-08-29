#ifndef HEAD_F32_H
#define HEAD_F32_H

#include <stdint.h>

void head_f32_zero(float *w, int vocab, int dim);
void head_f32_logits(float *logits, const float *w, const int8_t *x, int vocab, int dim);
void head_f32_train_step(float *w, const float *grad, const int8_t *x, int vocab, int dim, float lr);
float head_f32_mean_abs(const float *w, int n);

#endif