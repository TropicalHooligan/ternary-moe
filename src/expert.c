#include "expert.h"
#include "linear.h"
#include "act.h"

/*
 * ============================================================================
 * Expert Layer Utilities
 * ============================================================================
 */

/*
 * Compute row bytes for ternary packing.
 * Each row of dim elements requires (dim + 3) / 4 bytes.
 */

int expert_row_bytes(int in_dim) {
    return (in_dim + 3) / 4;
}

/*
 * Compute total bytes for W1 weights (hidden x dim).
 */

int expert_w1_bytes(int dim, int hidden) {
    return hidden * expert_row_bytes(dim);
}

/*
 * Compute total bytes for W2 weights (dim x hidden).
 */

int expert_w2_bytes(int dim, int hidden) {
    return dim * expert_row_bytes(hidden);
}

/*
 * ============================================================================
 * Expert Initialization
 * ============================================================================
 */

/*
 * Fill expert with a deterministic pattern.
 * W1: hidden x dim, W2: dim x hidden
 */

void expert_fill_pattern(uint8_t *w1, uint8_t *w2, int dim, int hidden) {
    tern_linear_fill_pattern(w1, hidden, dim);
    tern_linear_fill_pattern(w2, dim, hidden);
}

/*
 * Fill expert with a seeded deterministic pattern.
 */

void expert_fill_pattern_seed(uint8_t *w1, uint8_t *w2, int dim, int hidden, int seed) {
    tern_linear_fill_pattern_seed(w1, hidden, dim, seed);
    tern_linear_fill_pattern_seed(w2, dim, hidden, seed + 1);
}

/*
 * ============================================================================
 * Expert Forward Pass (Q8)
 * ============================================================================
 * 
 * Computes: out = ReLU(W2 @ ReLU(W1 @ x))
 * 
 * This is the main expert computation during inference.
 * 
 * Optimizations:
 * - Fused ReLU with linear operations
 * - Minimal memory allocations
 * - Reuses scratch buffer
 */

void expert_ffn_q8(int8_t *out,
                   const uint8_t *w1,
                   const uint8_t *w2,
                   const int8_t *x,
                   int dim,
                   int hidden,
                   int shift1,
                   int shift2,
                   int8_t *h,
                   int32_t *scratch) {
    /* First linear layer: h = W1 @ x */
    tern_linear_q8(h, w1, x, hidden, dim, shift1, scratch);
    
    /* ReLU activation */
    act_relu_q8(h, h, hidden);
    
    /* Second linear layer: out = W2 @ h */
    tern_linear_q8(out, w2, h, dim, hidden, shift2, scratch);
}
