#include "expert_random.h"
#include "ternary.h"

/*
 * ============================================================================
 * Random Expert Initialization
 * ============================================================================
 * 
 * Creates an expert with random ternary weights.
 * Uses a simple LCG for reproducibility.
 */

/* Simple LCG for random number generation */

static inline uint32_t expert_random_lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

/*
 * Fill a matrix with random ternary values.
 * 
 * Args:
 *   w - weight matrix (packed ternary)
 *   out_dim - number of output rows
 *   in_dim - number of input columns
 *   s - LCG state pointer
 */

static inline void fill_random_matrix(uint8_t *w, int out_dim, int in_dim, uint32_t *s) {
    int rb = (in_dim + 3) / 4;
    int bytes = out_dim * rb;
    
    /* Clear all weights */
    for (int b = 0; b < bytes; ++b) {
        w[b] = 0;
    }
    
    /* Fill with random ternary values */
    for (int o = 0; o < out_dim; ++o) {
        uint8_t *row = w + o * rb;
        
        for (int i = 0; i < in_dim; ++i) {
            uint32_t r = expert_random_lcg(s);
            int v = (int)((r >> 16) & 3u);
            
            /* Map: 0->0, 1->+1, 2->-1, 3->0 */
            if (v == 1) tern_set(row, i, 1);
            if (v == 2) tern_set(row, i, -1);
        }
    }
}

/*
 * Fill expert with random weights.
 * 
 * Args:
 *   w1 - W1 weights (hidden x dim)
 *   w2 - W2 weights (dim x hidden)
 *   dim - input/output dimension
 *   hidden - hidden dimension
 *   seed - random seed
 */

void expert_fill_random(uint8_t *w1, uint8_t *w2, int dim, int hidden, int seed) {
    /* Initialize LCG state */
    uint32_t s = (uint32_t)seed * 2654435761u + 12345u;
    
    /* Fill both weight matrices */
    fill_random_matrix(w1, hidden, dim, &s);
    fill_random_matrix(w2, dim, hidden, &s);
    
    /* Set diagonal elements to +1 for better initialization */
    int rb1 = (dim + 3) / 4;
    int rb2 = (hidden + 3) / 4;
    int m1 = hidden < dim ? hidden : dim;
    for (int o = 0; o < m1; ++o) {
        tern_set(w1 + o * rb1, o, 1);
    }
    int m2 = dim < hidden ? dim : hidden;
    for (int o = 0; o < m2; ++o) {
        tern_set(w2 + o * rb2, o, 1);
    }
}
