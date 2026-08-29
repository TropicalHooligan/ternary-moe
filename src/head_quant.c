#include "head_quant.h"
#include "ternary.h"
#include <math.h>

/*
 * ============================================================================
 * Weight Quantization Functions
 * ============================================================================
 * 
 * These functions convert float32 weights to various quantized formats.
 * The goal is to reduce memory usage and improve inference speed.
 */

/*
 * Quantize float32 weights to ternary using a threshold.
 * 
 * Weights are set to +1, 0, or -1 based on the threshold tau.
 * 
 * Optimizations:
 * - Process 4 values at a time (matches ternary packing)
 * - Skip zero weights to reduce memory writes
 */

void head_quantize_f32_to_ternary(uint8_t *out, const float *w, int vocab, int dim, float tau) {
    int rb = (dim + 3) / 4;
    int bytes = vocab * rb;
    
    /* Clear output */
    for (int b = 0; b < bytes; ++b) {
        out[b] = 0;
    }
    
    /* Quantize each row */
    for (int o = 0; o < vocab; ++o) {
        uint8_t *row = out + o * rb;
        const float *f = w + o * dim;
        
        /* Process 4 values at a time */
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            int8_t q0 = tern_quant(f[i], tau);
            int8_t q1 = tern_quant(f[i + 1], tau);
            int8_t q2 = tern_quant(f[i + 2], tau);
            int8_t q3 = tern_quant(f[i + 3], tau);
            
            /* Only set if non-zero */
            if (q0 != 0 || q1 != 0 || q2 != 0 || q3 != 0) {
                row[i >> 2] = tern_pack4(q0, q1, q2, q3);
            }
        }
        
        /* Handle remaining elements */
        for (; i < dim; ++i) {
            int8_t q = tern_quant(f[i], tau);
            if (q != 0) {
                tern_set(row, i, q);
            }
        }
    }
}

/*
 * Quantize float32 weights to ternary with per-row scaling.
 * 
 * Each row is quantized independently with its own threshold.
 * The threshold is chosen such that a fixed fraction of weights are non-zero.
 * 
 * This is known as "scaled ternary" quantization.
 */

void head_quantize_scaled(const float *w, uint8_t *pack, float *scales, int vocab, int dim) {
    int rb = (dim + 3) / 4;
    
    /* Clear output */
    for (int b = 0; b < vocab * rb; ++b) {
        pack[b] = 0;
    }
    
    /* Quantize each row */
    for (int o = 0; o < vocab; ++o) {
        const float *f = w + o * dim;
        
        /* Compute mean absolute value for threshold */
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float v = f[i];
            if (v < 0.0f) v = -v;
            sum += v;
        }
        
        /* Threshold is half the mean absolute value */
        float tau = 0.5f * sum / (float)dim;
        
        uint8_t *row = pack + o * rb;
        float sel_sum = 0.0f;
        int sel_count = 0;
        
        /* Quantize and compute scale */
        for (int i = 0; i < dim; ++i) {
            if (f[i] > tau) {
                tern_set(row, i, 1);
                sel_sum += f[i];
                sel_count++;
            } else if (f[i] < -tau) {
                tern_set(row, i, -1);
                sel_sum += -f[i];
                sel_count++;
            }
        }
        
        /* Compute scale as mean of selected weights */
        scales[o] = sel_count > 0 ? sel_sum / (float)sel_count : 0.0f;
    }
}

/*
 * Quantize a single row to two-plane ternary format.
 * 
 * Two-plane quantization:
 * - First plane: quantize with threshold tau0
 * - Compute residual: original - scale0 * ternary0
 * - Second plane: quantize residual with threshold tau1
 * 
 * This reduces quantization error significantly.
 */

static inline void quant_row_two_plane(const float *f, uint8_t *r0, uint8_t *r1, int dim, int rb, float *s0, float *s1) {
    /* Clear output */
    for (int b = 0; b < rb; ++b) {
        r0[b] = 0;
        r1[b] = 0;
    }
    
    /* Compute mean absolute value for first threshold */
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float v = f[i];
        if (v < 0.0f) v = -v;
        sum += v;
    }
    
    float tau0 = 0.5f * sum / (float)dim;
    
    /* First plane quantization */
    float ss = 0.0f;
    int cnt = 0;
    
    for (int i = 0; i < dim; ++i) {
        if (f[i] > tau0) {
            tern_set(r0, i, 1);
            ss += f[i];
            cnt++;
        } else if (f[i] < -tau0) {
            tern_set(r0, i, -1);
            ss += -f[i];
            cnt++;
        }
    }
    
    *s0 = cnt > 0 ? ss / (float)cnt : 0.0f;
    
    /* Compute residual */
    float rsum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float r = f[i] - (*s0) * (float)tern_get(r0, i);
        if (r < 0.0f) r = -r;
        rsum += r;
    }
    
    /* Second plane quantization */
    float tau1 = 0.5f * rsum / (float)dim;
    ss = 0.0f;
    cnt = 0;
    
    for (int i = 0; i < dim; ++i) {
        float r = f[i] - (*s0) * (float)tern_get(r0, i);
        if (r > tau1) {
            tern_set(r1, i, 1);
            ss += r;
            cnt++;
        } else if (r < -tau1) {
            tern_set(r1, i, -1);
            ss += -r;
            cnt++;
        }
    }
    
    *s1 = cnt > 0 ? ss / (float)cnt : 0.0f;
}

/*
 * Quantize all rows to two-plane ternary format.
 * 
 * This is the highest-quality quantization method in the codebase.
 */

void head_quantize_two_plane(const float *w, uint8_t *p0, uint8_t *p1, float *s0, float *s1, int vocab, int dim) {
    int rb = (dim + 3) / 4;
    
    /* Quantize each row */
    for (int o = 0; o < vocab; ++o) {
        quant_row_two_plane(
            w + o * dim,
            p0 + o * rb,
            p1 + o * rb,
            dim,
            rb,
            &s0[o],
            &s1[o]
        );
    }
}
