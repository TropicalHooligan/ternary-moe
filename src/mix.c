#include "mix.h"

/*
 * ============================================================================
 * Mixing Functions
 * ============================================================================
 * 
 * These functions combine outputs from multiple experts.
 */

/*
 * Average two Q8 vectors element-wise.
 * 
 * Computes: a = (a + b) / 2
 * 
 * Uses int16 accumulation to prevent overflow.
 */

void mix_avg_q8(int8_t *a, const int8_t *b, int n) {
    /* Process 4 values at a time */
    int i = 0;
    for (; i + 3 < n; i += 4) {
        int16_t v0 = (int16_t)a[i] + (int16_t)b[i];
        int16_t v1 = (int16_t)a[i + 1] + (int16_t)b[i + 1];
        int16_t v2 = (int16_t)a[i + 2] + (int16_t)b[i + 2];
        int16_t v3 = (int16_t)a[i + 3] + (int16_t)b[i + 3];
        
        a[i] = (int8_t)(v0 / 2);
        a[i + 1] = (int8_t)(v1 / 2);
        a[i + 2] = (int8_t)(v2 / 2);
        a[i + 3] = (int8_t)(v3 / 2);
    }
    
    /* Handle remaining elements */
    for (; i < n; ++i) {
        int16_t v = (int16_t)a[i] + (int16_t)b[i];
        a[i] = (int8_t)(v / 2);
    }
}
