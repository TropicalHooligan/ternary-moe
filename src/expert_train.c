#include "expert_train.h"
#include "ternary.h"

static int expert_grad_sign(float g) {
    if (g > 1e-6f) return 1;
    if (g < -1e-6f) return -1;
    return 0;
}

static int expert_acc_step(int16_t *acc, int idx, int s, int threshold) {
    if (s > 0) {
        acc[idx]++;
        if (acc[idx] >= threshold) {
            acc[idx] = 0;
            return 1;
        }
    } else if (s < 0) {
        acc[idx]--;
        if (acc[idx] <= -threshold) {
            acc[idx] = 0;
            return -1;
        }
    }
    return 0;
}

static float expert_grad_h_val(const uint8_t *w2, const float *grad_y, int dim, int rb2, int j) {
    float gh = 0.0f;
    for (int o = 0; o < dim; ++o) {
        gh += grad_y[o] * (float)tern_get(w2 + o * rb2, j);
    }
    return gh;
}

void expert_update_w2(uint8_t *w2, int16_t *acc2, const float *grad_y, const int8_t *h, int dim, int hidden, int threshold) {
    if (threshold >= 30000) return;
    if (threshold <= 0) threshold = 1;
    int rb = (hidden + 3) / 4;
    for (int o = 0; o < dim; ++o) {
        int gs = expert_grad_sign(grad_y[o]);
        if (gs == 0) continue;
        uint8_t *row = w2 + o * rb;
        int base = o * hidden;
        for (int j = 0; j < hidden; ++j) {
            int s = gs * h[j];
            int dir = expert_acc_step(acc2, base + j, -s, threshold);
            if (dir != 0) tern_adjust(row, j, dir);
        }
    }
}

void expert_update_w1(uint8_t *w1, int16_t *acc1, const uint8_t *w2, const float *grad_y, const int8_t *h, const int8_t *x, int dim, int hidden, int threshold) {
    if (threshold >= 30000) return;
    if (threshold <= 0) threshold = 1;
    int rb1 = (dim + 3) / 4;
    int rb2 = (hidden + 3) / 4;
    for (int j = 0; j < hidden; ++j) {
        if (h[j] <= 0) continue;
        float gh = expert_grad_h_val(w2, grad_y, dim, rb2, j);
        int gs = expert_grad_sign(gh);
        if (gs == 0) continue;
        uint8_t *row = w1 + j * rb1;
        int base = j * dim;
        for (int i = 0; i < dim; ++i) {
            int s = gs * x[i];
            int dir = expert_acc_step(acc1, base + i, -s, threshold);
            if (dir != 0) tern_adjust(row, i, dir);
        }
    }
}