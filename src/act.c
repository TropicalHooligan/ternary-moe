#include "act.h"

void act_relu_q8(int8_t *y, const int8_t *x, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}