#include "head_quant.h"
#include "ternary.h"

void head_quantize_f32_to_ternary(uint8_t *out, const float *w, int vocab, int dim, float tau) {
    int rb = (dim + 3) / 4;
    int bytes = vocab * rb;
    for (int b = 0; b < bytes; ++b) out[b] = 0;
    for (int o = 0; o < vocab; ++o) {
        uint8_t *row = out + o * rb;
        const float *f = w + o * dim;
        for (int i = 0; i < dim; ++i) {
            int8_t q = tern_quant(f[i], tau);
            if (q != 0) tern_set(row, i, q);
        }
    }
}

void head_quantize_scaled(const float *w, uint8_t *pack, float *scales, int vocab, int dim) {
    int rb = (dim + 3) / 4;
    for (int b = 0; b < vocab * rb; ++b) pack[b] = 0;
    for (int o = 0; o < vocab; ++o) {
        const float *f = w + o * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) { float v = f[i]; if (v < 0.0f) v = -v; sum += v; }
        float tau = 0.5f * sum / (float)dim;
        uint8_t *row = pack + o * rb;
        float sel_sum = 0.0f; int sel_count = 0;
        for (int i = 0; i < dim; ++i) {
            if (f[i] > tau) { tern_set(row, i, 1); sel_sum += f[i]; sel_count++; }
            else if (f[i] < -tau) { tern_set(row, i, -1); sel_sum += -f[i]; sel_count++; }
        }
        scales[o] = sel_count > 0 ? sel_sum / (float)sel_count : 0.0f;
    }
}

static void quant_row_two_plane(const float *f, uint8_t *r0, uint8_t *r1, int dim, int rb, float *s0, float *s1) {
    for (int b = 0; b < rb; ++b) { r0[b] = 0; r1[b] = 0; }
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) { float v = f[i]; if (v < 0.0f) v = -v; sum += v; }
    float tau0 = 0.5f * sum / (float)dim;
    float ss = 0.0f; int cnt = 0;
    for (int i = 0; i < dim; ++i) {
        if (f[i] > tau0) { tern_set(r0, i, 1); ss += f[i]; cnt++; }
        else if (f[i] < -tau0) { tern_set(r0, i, -1); ss += -f[i]; cnt++; }
    }
    *s0 = cnt > 0 ? ss / (float)cnt : 0.0f;
    float rsum = 0.0f;
    for (int i = 0; i < dim; ++i) { float r = f[i] - (*s0) * (float)tern_get(r0, i); if (r < 0.0f) r = -r; rsum += r; }
    float tau1 = 0.5f * rsum / (float)dim;
    ss = 0.0f; cnt = 0;
    for (int i = 0; i < dim; ++i) {
        float r = f[i] - (*s0) * (float)tern_get(r0, i);
        if (r > tau1) { tern_set(r1, i, 1); ss += r; cnt++; }
        else if (r < -tau1) { tern_set(r1, i, -1); ss += -r; cnt++; }
    }
    *s1 = cnt > 0 ? ss / (float)cnt : 0.0f;
}

void head_quantize_two_plane(const float *w, uint8_t *p0, uint8_t *p1, float *s0, float *s1, int vocab, int dim) {
    int rb = (dim + 3) / 4;
    for (int o = 0; o < vocab; ++o) {
        quant_row_two_plane(w + o * dim, p0 + o * rb, p1 + o * rb, dim, rb, &s0[o], &s1[o]);
    }
}