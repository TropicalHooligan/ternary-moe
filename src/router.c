#include "router.h"

int router_top1(const float *scores, int n) {
    if (n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (scores[i] > scores[best]) best = i;
    }
    return best;
}

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

    int a = 0;
    int b = 1;

    if (scores[b] > scores[a]) {
        a = 1;
        b = 0;
    }

    for (int i = 2; i < n; ++i) {
        if (scores[i] > scores[a]) {
            b = a;
            a = i;
        } else if (scores[i] > scores[b]) {
            b = i;
        }
    }

    *i0 = a;
    *i1 = b;
}