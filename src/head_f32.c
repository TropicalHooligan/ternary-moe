#include "head_f32.h"
#include <math.h>

/*
 * ============================================================================
 * Float32 Head Operations
 * ============================================================================
 * 
 * These are the reference implementations for the head layer.
 * Used during training when we need full precision.
 */

/*
 * Initialize head weights to zero.
 */
void head_f32_zero(float *w, int vocab, int dim) {
    int n = vocab * dim;
    
    /* Process 4 values at a time */
    int i = 0;
    for (; i + 3 < n; i += 4) {
        w[i] = 0.0f;
        w[i + 1] = 0.0f;
        w[i + 2] = 0.0f;
        w[i + 3] = 0.0f;
    }
    
    /* Handle remaining elements */
    for (; i < n; ++i) {
        w[i] = 0.0f;
    }
}

/*
 * Compute logits: logits = x @ W
 * 
 * This is a dense matrix-vector multiplication.
 * 
 * Optimizations:
 * - Loop reordering for better cache utilization
 * - Loop unrolling
 * - Use of local variables for accumulation
 */

void head_f32_logits(float *logits, const float *w, const int8_t *x, int vocab, int dim) {
    /* Convert int8 x to float for computation */
    float xf[32]; /* DIM is typically 32 */
    for (int i = 0; i < dim; ++i) {
        xf[i] = (float)x[i];
    }
    
    /* Compute logits for each output */
    for (int o = 0; o < vocab; ++o) {
        const float *row = w + o * dim;
        float sum = 0.0f;
        
        /* Unroll loop for better performance */
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            sum += row[i] * xf[i];
            sum += row[i + 1] * xf[i + 1];
            sum += row[i + 2] * xf[i + 2];
            sum += row[i + 3] * xf[i + 3];
        }
        
        /* Handle remaining elements */
        for (; i < dim; ++i) {
            sum += row[i] * xf[i];
        }
        
        logits[o] = sum;
    }
}

/*
 * Training step: W -= lr * grad @ x^T
 * 
 * This is the gradient descent update for the head weights.
 * 
 * Optimizations:
 * - Skip zero gradients
 * - Loop unrolling
 * - Weight clipping to [-1, 1] range
 */

void head_f32_train_step(float *w, const float *grad, const int8_t *x, int vocab, int dim, float lr) {
    /* Convert int8 x to float */
    float xf[32]; /* DIM is typically 32 */
    for (int i = 0; i < dim; ++i) {
        xf[i] = (float)x[i];
    }
    
    /* Update each row of W */
    for (int o = 0; o < vocab; ++o) {
        float g = grad[o];
        
        /* Skip if gradient is zero */
        if (g == 0.0f) continue;
        
        float *row = w + o * dim;
        float lr_g = lr * g;
        
        /* Update each weight */
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            row[i] -= lr_g * xf[i];
            row[i + 1] -= lr_g * xf[i + 1];
            row[i + 2] -= lr_g * xf[i + 2];
            row[i + 3] -= lr_g * xf[i + 3];
        }
        
        /* Handle remaining elements */
        for (; i < dim; ++i) {
            row[i] -= lr_g * xf[i];
        }
        
        /* Clip weights to [-1, 1] range */
        for (int i = 0; i < dim; ++i) {
            if (row[i] > 1.0f) row[i] = 1.0f;
            else if (row[i] < -1.0f) row[i] = -1.0f;
        }
    }
}

/*
 * Compute mean absolute value of weights.
 * Used for monitoring during training.
 */

float head_f32_mean_abs(const float *w, int n) {
    if (n <= 0) return 0.0f;
    
    float sum = 0.0f;
    
    /* Process 4 values at a time */
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float v0 = w[i];
        float v1 = w[i + 1];
        float v2 = w[i + 2];
        float v3 = w[i + 3];
        
        v0 = v0 < 0.0f ? -v0 : v0;
        v1 = v1 < 0.0f ? -v1 : v1;
        v2 = v2 < 0.0f ? -v2 : v2;
        v3 = v3 < 0.0f ? -v3 : v3;
        
        sum += v0 + v1 + v2 + v3;
    }
    
    /* Handle remaining elements */
    for (; i < n; ++i) {
        float v = w[i];
        if (v < 0.0f) v = -v;
        sum += v;
    }
    
    return sum / (float)n;
}
