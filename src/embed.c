#include "embed.h"

void embed_fill_pattern(int8_t *emb, int vocab, int dim) {
    for (int t = 0; t < vocab; ++t) {
        int8_t *row = emb + t * dim;
        for (int i = 0; i < dim; ++i) {
            row[i] = ((t + i) & 1) ? 16 : -16;
        }
    }
}

void embed_fill_hash(int8_t *emb, int vocab, int dim, int scale) {
    if (scale > 127) scale = 127;
    if (scale < 0) scale = 0;
    for (int t = 0; t < vocab; ++t) {
        int8_t *row = emb + t * dim;
        uint32_t s = (uint32_t)t * 2654435761u + 12345u;
        for (int i = 0; i < dim; ++i) {
            s = s * 1664525u + 1013904223u;
            row[i] = (int8_t)(((s >> 16) & 1u) ? scale : -scale);
        }
    }
}

const int8_t* embed_row(const int8_t *emb, int token, int dim) {
    return emb + token * dim;
}

void embed_add_byte(int32_t *acc, const int8_t *emb, int token, int dim) {
    const int8_t *row = embed_row(emb, token, dim);
    for (int i = 0; i < dim; ++i) {
        acc[i] += row[i];
    }
}

void embed_acc_to_q8(int8_t *out, const int32_t *acc, int dim, int count) {
    if (count <= 0) {
        for (int i = 0; i < dim; ++i) out[i] = 0;
        return;
    }
    for (int i = 0; i < dim; ++i) {
        int32_t v = acc[i] / count;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }
}