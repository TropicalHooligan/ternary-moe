#include "act.h"

/*
 * ============================================================================
 * ReLU Activation Function (Q8)
 * ============================================================================
 * 
 * ReLU: f(x) = max(0, x)
 * 
 * Optimized implementation for int8 inputs.
 * 
 * Optimizations:
 * - Loop unrolling for better pipelining
 * - Branchless comparison where beneficial
 * - Process 4 values at a time
 */

void act_relu_q8(int8_t *y, const int8_t *x, int n) {
    int i = 0;
    
    /* Process 4 values at a time */
    for (; i + 3 < n; i += 4) {
        /* Use ternary operator for clarity and good code generation */
        y[i] = x[i] > 0 ? x[i] : 0;
        y[i + 1] = x[i + 1] > 0 ? x[i + 1] : 0;
        y[i + 2] = x[i + 2] > 0 ? x[i + 2] : 0;
        y[i + 3] = x[i + 3] > 0 ? x[i + 3] : 0;
    }
    
    /* Handle remaining elements */
    for (; i < n; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}
