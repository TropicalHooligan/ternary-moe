#include "loss.h"
#include <math.h>

/*
 * ============================================================================
 * Softmax and Cross-Entropy Loss
 * ============================================================================
 * 
 * These functions compute the loss and gradients for training.
 */

/*
 * Compute softmax from logits.
 * 
 * Args:
 *   p - output probabilities (normalized, sum to 1)
 *   logits - input logits
 *   n - number of elements
 * 
 * Uses numerical stability trick: subtract max before exp.
 */

void softmax_from_logits_f32(float *p, const int32_t *logits, int n) {
    if (n <= 0) return;
    
    /* Find max for numerical stability */
    float m = (float)logits[0];
    for (int i = 1; i < n; ++i) {
        float v = (float)logits[i];
        if (v > m) m = v;
    }
    
    /* Compute exp(logit - max) and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        p[i] = expf((float)logits[i] - m);
        sum += p[i];
    }
    
    /* Normalize */
    float inv = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < n; ++i) {
        p[i] *= inv;
    }
}

/*
 * Compute cross-entropy loss for a single target.
 * 
 * Args:
 *   p - probabilities (from softmax)
 *   target - index of target class
 *   n - number of classes
 * 
 * Returns: -log(p[target])
 */

float ce_loss_f32(const float *p, int target, int n) {
    if (n <= 0) return 0.0f;
    
    /* Clamp target to valid range */
    if (target < 0 || target >= n) target = 0;
    
    /* Clamp probability to prevent log(0) */
    float q = p[target];
    if (q < 1e-12f) q = 1e-12f;
    
    return -logf(q);
}

/*
 * Compute cross-entropy gradient.
 * 
 * Args:
 *   grad - output gradient (same size as p)
 *   p - probabilities (from softmax)
 *   target - index of target class
 *   n - number of classes
 * 
 * For cross-entropy loss with softmax, the gradient is:
 *   grad[i] = p[i] - (1 if i == target else 0)
 */

void softmax_ce_grad_f32(float *grad, const float *p, int target, int n) {
    if (n <= 0) return;
    
    /* Clamp target */
    if (target < 0 || target >= n) target = 0;
    
    /* grad[i] = p[i] for all i */
    for (int i = 0; i < n; ++i) {
        grad[i] = p[i];
    }
    
    /* grad[target] -= 1 */
    grad[target] -= 1.0f;
}

/*
 * Combined softmax, cross-entropy loss, and gradient computation.
 * 
 * This is the most common operation during training.
 * 
 * Args:
 *   grad - output gradient
 *   logits - input logits
 *   target - target class index
 *   n - number of classes
 * 
 * Returns: cross-entropy loss
 */

float softmax_ce_loss_and_grad_f32(float *grad, const int32_t *logits, int target, int n) {
    /* Compute softmax */
    softmax_from_logits_f32(grad, logits, n);
    
    /* Compute loss */
    float loss = ce_loss_f32(grad, target, n);
    
    /* Compute gradient */
    softmax_ce_grad_f32(grad, grad, target, n);
    
    return loss;
}
