#ifndef LOSS_H
#define LOSS_H

#include <stdint.h>

void softmax_from_logits_f32(float *p, const int32_t *logits, int n);
float ce_loss_f32(const float *p, int target, int n);
void softmax_ce_grad_f32(float *grad, const float *p, int target, int n);
float softmax_ce_loss_and_grad_f32(float *grad, const int32_t *logits, int target, int n);

#endif