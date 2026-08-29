#include "moe.h"
#include "expert.h"
#include "router.h"
#include "mix.h"

/*
 * ============================================================================
 * Mixture of Experts Utilities
 * ============================================================================
 */

/*
 * Compute total bytes for all W1 weights across all experts.
 */

int moe_w1_total_bytes(int experts, int dim, int hidden) {
    return experts * expert_w1_bytes(dim, hidden);
}

/*
 * Compute total bytes for all W2 weights across all experts.
 */

int moe_w2_total_bytes(int experts, int dim, int hidden) {
    return experts * expert_w2_bytes(dim, hidden);
}

/*
 * ============================================================================
 * MoE Initialization
 * ============================================================================
 */

/*
 * Fill all experts with deterministic patterns.
 * Each expert gets a unique seed for pattern generation.
 */

void moe_fill_pattern(uint8_t *w1_all, uint8_t *w2_all, int experts, int dim, int hidden) {
    int wb1 = expert_w1_bytes(dim, hidden);
    int wb2 = expert_w2_bytes(dim, hidden);
    
    for (int e = 0; e < experts; ++e) {
        expert_fill_pattern_seed(
            w1_all + e * wb1,
            w2_all + e * wb2,
            dim,
            hidden,
            e * 5 + 1
        );
    }
}

/*
 * ============================================================================
 * MoE Forward Pass
 * ============================================================================
 */

/*
 * Forward pass through top-1 expert only.
 * 
 * This is the most common use case - route to the single best expert.
 * 
 * Args:
 *   out - output buffer (dim elements)
 *   w1_all - all W1 weights concatenated
 *   w2_all - all W2 weights concatenated
 *   scores - router scores for each expert
 *   x - input vector (dim elements)
 *   experts - number of experts
 *   dim - input/output dimension
 *   hidden - hidden dimension
 *   shift1 - shift for first linear layer
 *   shift2 - shift for second linear layer
 *   h - temporary buffer for hidden layer (hidden elements)
 *   scratch - temporary buffer for accumulation (max(dim, hidden) elements)
 */

void moe_forward_top1(int8_t *out,
                      const uint8_t *w1_all,
                      const uint8_t *w2_all,
                      const float *scores,
                      const int8_t *x,
                      int experts,
                      int dim,
                      int hidden,
                      int shift1,
                      int shift2,
                      int8_t *h,
                      int32_t *scratch) {
    if (experts <= 0) {
        /* No experts: zero output */
        for (int i = 0; i < dim; ++i) {
            out[i] = 0;
        }
        return;
    }
    
    /* Find best expert */
    int e = router_top1(scores, experts);
    
    /* Get weights for this expert */
    int wb1 = expert_w1_bytes(dim, hidden);
    int wb2 = expert_w2_bytes(dim, hidden);
    const uint8_t *w1 = w1_all + e * wb1;
    const uint8_t *w2 = w2_all + e * wb2;
    
    /* Run expert */
    expert_ffn_q8(out, w1, w2, x, dim, hidden, shift1, shift2, h, scratch);
}

/*
 * Forward pass through top-2 experts with averaging.
 * 
 * This uses two experts and averages their outputs.
 * 
 * Args:
 *   out - output buffer (dim elements)
 *   tmp - temporary buffer for second expert output (dim elements)
 *   ... (same as moe_forward_top1)
 *   h1, h2 - temporary buffers for hidden layers
 *   scratch - temporary buffer for accumulation
 */

void moe_forward_top2(int8_t *out,
                      int8_t *tmp,
                      const uint8_t *w1_all,
                      const uint8_t *w2_all,
                      const float *scores,
                      const int8_t *x,
                      int experts,
                      int dim,
                      int hidden,
                      int shift1,
                      int shift2,
                      int8_t *h1,
                      int8_t *h2,
                      int32_t *scratch) {
    if (experts <= 0) {
        /* No experts: zero output */
        for (int i = 0; i < dim; ++i) {
            out[i] = 0;
        }
        return;
    }
    
    /* Find top-2 experts */
    int e0, e1;
    router_top2(scores, experts, &e0, &e1);
    
    /* Get weights */
    int wb1 = expert_w1_bytes(dim, hidden);
    int wb2 = expert_w2_bytes(dim, hidden);
    
    /* Run first expert */
    expert_ffn_q8(out, w1_all + e0 * wb1, w2_all + e0 * wb2, x, dim, hidden, shift1, shift2, h1, scratch);
    
    /* If both experts are the same, we're done */
    if (e1 == e0) return;
    
    /* Run second expert */
    expert_ffn_q8(tmp, w1_all + e1 * wb1, w2_all + e1 * wb2, x, dim, hidden, shift1, shift2, h2, scratch);
    
    /* Average outputs */
    mix_avg_q8(out, tmp, dim);
}
