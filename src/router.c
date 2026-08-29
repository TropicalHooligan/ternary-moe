#include "router.h"

/*
 * ============================================================================
 * Router Functions
 * ============================================================================
 * 
 * These functions implement the routing logic for MoE layers.
 * They select which expert(s) to use based on the scores.
 */

/*
 * Find the expert with the highest score.
 * 
 * Returns: index of best expert, or -1 if no experts
 */

int router_top1(const float *scores, int n) {
    if (n <= 0) return -1;
    
    int best = 0;
    float best_score = scores[0];
    
    /* Linear scan for maximum */
    for (int i = 1; i < n; ++i) {
        if (scores[i] > best_score) {
            best = i;
            best_score = scores[i];
        }
    }
    
    return best;
}

/*
 * Find the top-2 experts by score.
 * 
 * Returns: indices of top-2 experts in i0 (best) and i1 (second best)
 */

void router_top2(const float *scores, int n, int *i0, int *i1) {
    if (n <= 0) {
        *i0 = -1;
        *i1 = -1;
        return;
    }
    
    if (n == 1) {
        *i0 = 0;
        *i1 = 0;
        return;
    }
    
    /* Initialize with first two elements */
    int a = 0;
    int b = 1;
    
    if (scores[b] > scores[a]) {
        a = 1;
        b = 0;
    }
    
    /* Scan remaining elements */
    for (int i = 2; i < n; ++i) {
        if (scores[i] > scores[a]) {
            /* New best: shift down */
            b = a;
            a = i;
        } else if (scores[i] > scores[b]) {
            /* New second best */
            b = i;
        }
    }
    
    *i0 = a;
    *i1 = b;
}
