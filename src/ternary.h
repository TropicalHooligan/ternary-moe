#ifndef TERNARY_H
#define TERNARY_H

#include <stdint.h>
#include <stddef.h>

/*
 * Ternary weight packing: 4 values per byte, 2 bits each
 * Encoding: 00=0, 01=+1, 10=-1, 11=0 (unused)
 * 
 * Memory layout optimization:
 * - Weights stored as packed bytes (4 ternary values per byte)
 * - Access pattern optimized for sequential memory access
 * - Dot product unrolled for better pipelining
 */

/* Lookup table for backward compatibility (deprecated in hot paths) */
extern const int8_t TERN_LUT[4];

/*
 * Pack 4 ternary values into a single byte.
 * Each value uses 2 bits: 00=0, 01=+1, 10=-1, 11=0
 */
static inline uint8_t tern_pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    uint8_t code = 0;
    code |= (a > 0 ? 1u : (a < 0 ? 2u : 0u));
    code |= (b > 0 ? 1u : (b < 0 ? 2u : 0u)) << 2;
    code |= (c > 0 ? 1u : (c < 0 ? 2u : 0u)) << 4;
    code |= (d > 0 ? 1u : (d < 0 ? 2u : 0u)) << 6;
    return code;
}

/*
 * Get ternary value at index i from packed array.
 * This is the fallback for non-aligned access.
 */
static inline int8_t tern_get(const uint8_t *p, int i) {
    unsigned code = (p[i >> 2] >> (2 * (i & 3))) & 3u;
    /* Fast decode: (code & 1) - ((code >> 1) & 1) */
    return (int8_t)((code & 1u) - ((code >> 1) & 1u));
}

/*
 * Quantize float to ternary: +1, 0, or -1
 * Uses fast comparison without branching when possible.
 */
static inline int8_t tern_quant(float w, float tau) {
    /* Fast path: avoid function calls, use direct comparisons */
    if (w > tau) return 1;
    if (w < -tau) return -1;
    return 0;
}

/*
 * Optimized ternary dot product.
 * This is the HOT PATH - called millions of times during training.
 * 
 * Optimizations:
 * - Loop unrolled by 4 (processes 4 values per packed byte)
 * - Uses arithmetic instead of LUT for decoding (faster, no cache miss)
 * - Full bytes processed first, then tail
 * - Uses restrict to enable better aliasing
 * - Minimizes pointer arithmetic in inner loop
 * 
 * Args:
 *   x - input vector (int8_t), length n
 *   w - packed ternary weights (uint8_t), n/4 bytes for full n
 *   n - length of both vectors (must be multiple of 4 for best performance)
 * 
 * Returns: dot product as int32_t
 */
static inline int32_t tern_dot(const int8_t *restrict x, const uint8_t *restrict w, int n) {
    int32_t sum = 0;
    int i = 0;
    
    /* Process full bytes (4 values at a time) */
    const int full_bytes = n >> 2;
    
    /* Prefetch first byte */
    for (int b = 0; b < full_bytes; ++b) {
        unsigned byte = w[b];
        /* Decode all 4 lanes using bit manipulation */
        /* Lane 0: bits 0-1 */
        sum += (int32_t)x[i]     * (int32_t)((byte & 1u) - ((byte >> 1) & 1u));
        /* Lane 1: bits 2-3 */
        sum += (int32_t)x[i + 1] * (int32_t)(((byte >> 2) & 1u) - ((byte >> 3) & 1u));
        /* Lane 2: bits 4-5 */
        sum += (int32_t)x[i + 2] * (int32_t)(((byte >> 4) & 1u) - ((byte >> 5) & 1u));
        /* Lane 3: bits 6-7 */
        sum += (int32_t)x[i + 3] * (int32_t)(((byte >> 6) & 1u) - ((byte >> 7) & 1u));
        i += 4;
    }
    
    /* Process remaining elements (tail) */
    for (; i < n; ++i) {
        unsigned code = (w[i >> 2] >> (2 * (i & 3))) & 3u;
        int8_t val = (int8_t)((code & 1u) - ((code >> 1) & 1u));
        sum += (int32_t)x[i] * (int32_t)val;
    }
    
    return sum;
}

/*
 * Set ternary value at index i in packed array.
 * Used during weight updates.
 */
static inline void tern_set(uint8_t *p, int i, int8_t v) {
    uint8_t code = (v > 0) ? 1u : ((v < 0) ? 2u : 0u);
    int shift = 2 * (i & 3);
    uint8_t mask = (uint8_t)(3u << shift);
    p[i >> 2] = (uint8_t)((p[i >> 2] & ~mask) | (code << shift));
}

/*
 * Adjust ternary value at index i by direction.
 * Used during training for weight updates.
 * 
 * Args:
 *   p - packed array
 *   i - index
 *   dir - +1 to increase, -1 to decrease
 */
static inline void tern_adjust(uint8_t *p, int i, int dir) {
    int8_t v = tern_get(p, i);
    if (dir > 0 && v < 1) v++;
    else if (dir < 0 && v > -1) v--;
    tern_set(p, i, v);
}

/*
 * Batch tern_set for setting multiple values in a row.
 * More efficient when setting consecutive indices.
 */
static inline void tern_set4(uint8_t *p, int i, int8_t a, int8_t b, int8_t c, int8_t d) {
    uint8_t code = tern_pack4(a, b, c, d);
    p[i >> 2] = code;
}

/*
 * Get 4 consecutive ternary values from packed array.
 * Used for SIMD-friendly access patterns.
 */
static inline void tern_get4(const uint8_t *p, int i, int8_t *out) {
    unsigned byte = p[i >> 2];
    out[0] = (int8_t)((byte & 1u) - ((byte >> 1) & 1u));
    out[1] = (int8_t)(((byte >> 2) & 1u) - ((byte >> 3) & 1u));
    out[2] = (int8_t)(((byte >> 4) & 1u) - ((byte >> 5) & 1u));
    out[3] = (int8_t)(((byte >> 6) & 1u) - ((byte >> 7) & 1u));
}

#endif /* TERNARY_H */