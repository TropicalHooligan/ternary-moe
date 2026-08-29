#include "head_f32.h"

void head_f32_zero(float *w, int vocab, int dim) {
    int n = vocab * dim;
    for (int i = 0; i < n; ++i) w[i] = 0.0f;
}

void head_f32_logits(float *logits, const float *w, const int8_t *x, int vocab, int dim) {
    for (int o = 0; o < vocab; ++o) {
        const float *row = w + o * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum += row[i] * (float)x[i];
        }
        logits[o] = sum;
    }
}

void head_f32_train_step(float *w, const float *grad, const int8_t *x, int vocab, int dim, float lr) {
    for (int o = 0; o < vocab; ++o) {
        float g = grad[o];
        if (g == 0.0f) continue;
        float *row = w + o * dim;
        for (int i = 0; i < dim; ++i) {
            float v = row[i] - lr * g * (float)x[i];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            row[i] = v;
        }
    }
}

float head_f32_mean_abs(const float *w, int n) {
    if (n <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = w[i];
        if (v < 0.0f) v = -v;
        sum += v;
    }
    return sum / (float)n;
}