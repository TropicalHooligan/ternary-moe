#include "loss.h"
#include <math.h>

void softmax_from_logits_f32(float *p, const int32_t *logits, int n) {
    if (n <= 0) return;

    float m = (float)logits[0];
    for (int i = 1; i < n; ++i) {
        if ((float)logits[i] > m) m = (float)logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        p[i] = expf((float)logits[i] - m);
        sum += p[i];
    }

    float inv = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < n; ++i) p[i] *= inv;
}

float ce_loss_f32(const float *p, int target, int n) {
    if (n <= 0) return 0.0f;
    if (target < 0 || target >= n) target = 0;

    float q = p[target];
    if (q < 1e-12f) q = 1e-12f;

    return -logf(q);
}

void softmax_ce_grad_f32(float *grad, const float *p, int target, int n) {
    if (n <= 0) return;
    if (target < 0 || target >= n) target = 0;

    for (int i = 0; i < n; ++i) grad[i] = p[i];
    grad[target] -= 1.0f;
}

float softmax_ce_loss_and_grad_f32(float *grad, const int32_t *logits, int target, int n) {
    softmax_from_logits_f32(grad, logits, n);
    float loss = ce_loss_f32(grad, target, n);
    softmax_ce_grad_f32(grad, grad, target, n);
    return loss;
}