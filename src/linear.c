#include "linear.h"
#include "ternary.h"

/*
 * ============================================================================
 * Pattern Generation for Weight Initialization
 * ============================================================================
 * 
 * Creates deterministic pseudo-random ternary patterns for weight matrices.
 * This is used for reproducible initialization.
 */

static inline int8_t pattern_val(int o, int i, int in_dim) {
    if (i >= in_dim) return 0;
    /* Simple checkerboard pattern: +1 or -1 based on (o+i) parity */
    return ((o + i) & 1) ? 1 : -1;
}

static inline int8_t pattern_val_seed(int seed, int o, int i, int in_dim) {
    if (i >= in_dim) return 0;
    /* More complex pattern using hash */
    int v = (seed * 7 + o * 3 + i) & 3;
    if (v == 0) return 0;
    if (v == 1) return 1;
    return -1;
}

/*
 * Fill a ternary weight matrix with a deterministic pattern.
 * Each row has a unique pattern based on its index.
 */
void tern_linear_fill_pattern(uint8_t *w, int out_dim, int in_dim) {
    int rb = (in_dim + 3) / 4;
    
    for (int o = 0; o < out_dim; ++o) {
        uint8_t *row = w + o * rb;
        
        /* Process 4 values at a time */
        for (int b = 0; b < rb; ++b) {
            int i = b * 4;
            row[b] = tern_pack4(
                pattern_val(o, i, in_dim),
                pattern_val(o, i + 1, in_dim),
                pattern_val(o, i + 2, in_dim),
                pattern_val(o, i + 3, in_dim)
            );
        }
    }
}

/*
 * Fill a ternary weight matrix with a seeded deterministic pattern.
 * Uses a hash-based pattern for better randomness.
 */
void tern_linear_fill_pattern_seed(uint8_t *w, int out_dim, int in_dim, int seed) {
    int rb = (in_dim + 3) / 4;
    
    for (int o = 0; o < out_dim; ++o) {
        uint8_t *row = w + o * rb;
        
        for (int b = 0; b < rb; ++b) {
            int i = b * 4;
            row[b] = tern_pack4(
                pattern_val_seed(seed, o, i, in_dim),
                pattern_val_seed(seed, o, i + 1, in_dim),
                pattern_val_seed(seed, o, i + 2, in_dim),
                pattern_val_seed(seed, o, i + 3, in_dim)
            );
        }
    }
}

/*
 * ============================================================================
 * Ternary Linear Layer (int32 accumulation)
 * ============================================================================
 * 
 * Computes: y = x @ W where W is ternary
 * 
 * This is the core computation - optimized for speed.
 * Uses int32 accumulation for precision, then converts to int8.
 */

void tern_linear_i32(int32_t *y, const uint8_t *w, const int8_t *x, int out_dim, int in_dim) {
    int rb = (in_dim + 3) / 4;
    
    /* Process each output dimension */
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = w + o * rb;
        
        /* Use tern_dot for the actual computation */
        y[o] = tern_dot(x, row, in_dim);
    }
}

/*
 * ============================================================================
 * Quantization to Q8 (int8)
 * ============================================================================
 * 
 * Converts int32 accumulations to int8 with scaling.
 * Includes saturation to prevent overflow.
 */

void linear_i32_to_q8(int8_t *y, const int32_t *acc, int n, int shift) {
    /* Process 4 values at a time for better pipelining */
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int32_t v0 = acc[i] >> shift;
        int32_t v1 = acc[i + 1] >> shift;
        int32_t v2 = acc[i + 2] >> shift;
        int32_t v3 = acc[i + 3] >> shift;
        
        /* Saturate to int8 range */
        v0 = v0 > 127 ? 127 : (v0 < -127 ? -127 : v0);
        v1 = v1 > 127 ? 127 : (v1 < -127 ? -127 : v1);
        v2 = v2 > 127 ? 127 : (v2 < -127 ? -127 : v2);
        v3 = v3 > 127 ? 127 : (v3 < -127 ? -127 : v3);
        
        y[i] = (int8_t)v0;
        y[i + 1] = (int8_t)v1;
        y[i + 2] = (int8_t)v2;
        y[i + 3] = (int8_t)v3;
    }
    
    /* Handle remaining elements */
    for (; i < n; ++i) {
        int32_t v = acc[i] >> shift;
        if (v > 127) v = 127;
        else if (v < -127) v = -127;
        y[i] = (int8_t)v;
    }
}

/*
 * ============================================================================
 * Ternary Linear Layer (Q8 output)
 * ============================================================================
 * 
 * Combined operation: y = W @ x, with int32 accumulation and Q8 output.
 * 
 * This is the most common use case in the network.
 */

void tern_linear_q8(int8_t *y, const uint8_t *w, const int8_t *x, int out_dim, int in_dim, int shift, int32_t *scratch) {
    /* Compute with int32 precision */
    tern_linear_i32(scratch, w, x, out_dim, in_dim);
    
    /* Convert to Q8 */
    linear_i32_to_q8(y, scratch, out_dim, shift);
}
