#include "context.h"
#include "embed.h"

/*
 * ============================================================================
 * Context Vector Computation
 * ============================================================================
 * 
 * Computes the context vector from a sliding window of tokens.
 * This is the input to the MoE layer.
 */

/*
 * Compute context vector from token window.
 * 
 * Args:
 *   out - output context vector (dim elements, Q8)
 *   acc - accumulation buffer (dim elements, int32)
 *   w - token window
 *   emb - embedding matrix (vocab x dim, Q8)
 *   dim - embedding dimension
 *   ctx - context length (number of tokens to include)
 * 
 * The context vector is the average of the embeddings of the last ctx tokens.
 */

void context_vector_from_window(int8_t *out, int32_t *acc, const ByteWindow *w, const int8_t *emb, int dim, int ctx) {
    /* Get number of tokens in window */
    int n = window_len(w);
    if (ctx < n) n = ctx;
    
    /* Clear accumulator */
    for (int i = 0; i < dim; ++i) {
        acc[i] = 0;
    }
    
    /* Sum embeddings of tokens in window */
    for (int age = 0; age < n; ++age) {
        uint8_t token = window_get(w, age);
        embed_add_byte(acc, emb, token, dim);
    }
    
    /* Average and convert to Q8 */
    embed_acc_to_q8(out, acc, dim, n);
}
