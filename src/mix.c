#include "mix.h"

void mix_avg_q8(int8_t *a, const int8_t *b, int n) {
    for (int i = 0; i < n; ++i) {
        int16_t v = (int16_t)a[i] + (int16_t)b[i];
        a[i] = (int8_t)(v / 2);
    }
}