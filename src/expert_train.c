#include "expert_train.h"
#include "ternary.h"

/*
 * ============================================================================
 * Expert Training Utilities
 * ============================================================================
 * 
 * These functions implement ternary weight updates for expert layers.
 * The key insight is that we can't directly update ternary weights with
 * gradient descent, so we use a threshold-based approach.
 */

/*
 * Get sign of gradient with thresholding.
 * 
 * Args:
 *   g - gradient value
 * 
 * Returns: +1 if g > threshold, -1 if g < -threshold, 0 otherwise
 */

static inline int expert_grad_sign(float g) {
    /* Use a small threshold to avoid jitter from tiny gradients */
    if (g > 1e-6f) return 1;
    if (g < -1e-6f) return -1;
    return 0;
}

/*
 * Update accumulation buffer and check if threshold is crossed.
 * 
 * This implements a simple "stochastic threshold" approach:
 * - Accumulate gradient sign * learning_rate
 * - When accumulation crosses threshold, flip the weight
 * - Reset accumulation
 * 
 * Args:
 *   acc - accumulation buffer
 *   idx - index in buffer
 *   s - sign of gradient contribution (+1 or -1)
 *   threshold - threshold for weight flip
 * 
 * Returns: +1 if weight should be increased, -1 if decreased, 0 otherwise
 */

static inline int expert_acc_step(int16_t *acc, int idx, int s, int threshold) {
    if (s > 0) {
        acc[idx]++;
        if (acc[idx] >= threshold) {
            acc[idx] = 0;
            return 1;
        }
    } else if (s < 0) {
        acc[idx]--;
        if (acc[idx] <= -threshold) {
            acc[idx] = 0;
            return -1;
        }
    }
    return 0;
}

/*
 * Compute gradient for hidden layer.
 * 
 * Args:
 *   w2 - second layer weights
 *   grad_y - gradient from output
 *   dim - output dimension
 *   rb2 - row bytes for w2
 *   j - hidden dimension index
 * 
 * Returns: gradient for hidden layer at index j
 */

static inline float expert_grad_h_val(const uint8_t *w2, const float *grad_y, int dim, int rb2, int j) {
    float gh = 0.0f;
    
    /* Compute gradient contribution from each output */
    for (int o = 0; o < dim; ++o) {
        gh += grad_y[o] * (float)tern_get(w2 + o * rb2, j);
    }
    
    return gh;
}

/*
 * Update W2 weights based on gradient.
 * 
 * Args:
 *   w2 - W2 weights (dim x hidden, packed ternary)
 *   acc2 - accumulation buffer for W2
 *   grad_y - gradient from output (dim elements)
 *   h - hidden layer activations (hidden elements)
 *   dim - output dimension
 *   hidden - hidden dimension
 *   threshold - update threshold
 */

void expert_update_w2(uint8_t *w2, int16_t *acc2, const float *grad_y, const int8_t *h, int dim, int hidden, int threshold) {
    /* Safety check */
    if (threshold >= 30000) return;
    if (threshold <= 0) threshold = 1;
    
    int rb = (hidden + 3) / 4;
    
    /* Process each output dimension */
    for (int o = 0; o < dim; ++o) {
        int gs = expert_grad_sign(grad_y[o]);
        if (gs == 0) continue;
        
        /* Get W2 row for this output */
        uint8_t *row = w2 + o * rb;
        int base = o * hidden;
        
        /* Update each weight in the row */
        for (int j = 0; j < hidden; ++j) {
            /* Gradient contribution: gs * h[j] */
            int s = gs * h[j];
            
            /* Update accumulation and check threshold */
            int dir = expert_acc_step(acc2, base + j, -s, threshold);
            
            /* If threshold crossed, adjust weight */
            if (dir != 0) {
                tern_adjust(row, j, dir);
            }
        }
    }
}

/*
 * Update W1 weights based on gradient.
 * 
 * Args:
 *   w1 - W1 weights (hidden x dim, packed ternary)
 *   acc1 - accumulation buffer for W1
 *   w2 - W2 weights (for computing hidden gradient)
 *   grad_y - gradient from output (dim elements)
 *   h - hidden layer activations (hidden elements)
 *   x - input vector (dim elements)
 *   dim - input/output dimension
 *   hidden - hidden dimension
 *   threshold - update threshold
 */

void expert_update_w1(uint8_t *w1, int16_t *acc1, const uint8_t *w2, const float *grad_y, const int8_t *h, const int8_t *x, int dim, int hidden, int threshold) {
    /* Safety check */
    if (threshold >= 30000) return;
    if (threshold <= 0) threshold = 1;
    
    int rb1 = (dim + 3) / 4;
    int rb2 = (hidden + 3) / 4;
    
    /* Process each hidden dimension */
    for (int j = 0; j < hidden; ++j) {
        /* Skip if hidden activation is zero (ReLU gradient) */
        if (h[j] <= 0) continue;
        
        /* Compute gradient for hidden layer at j */
        float gh = expert_grad_h_val(w2, grad_y, dim, rb2, j);
        int gs = expert_grad_sign(gh);
        if (gs == 0) continue;
        
        /* Get W1 row for this hidden dimension */
        uint8_t *row = w1 + j * rb1;
        int base = j * dim;
        
        /* Update each weight in the row */
        for (int i = 0; i < dim; ++i) {
            /* Gradient contribution: gs * x[i] */
            int s = gs * x[i];
            
            /* Update accumulation and check threshold */
            int dir = expert_acc_step(acc1, base + i, -s, threshold);
            
            /* If threshold crossed, adjust weight */
            if (dir != 0) {
                tern_adjust(row, i, dir);
            }
        }
    }
}
