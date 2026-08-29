#include "context.h"
#include "embed.h"

void context_vector_from_window(int8_t *out, int32_t *acc, const ByteWindow *w, const int8_t *emb, int dim, int ctx) {
    int n = window_len(w);
    if (ctx < n) n = ctx;
    for (int i = 0; i < dim; ++i) acc[i] = 0;
    for (int age = 0; age < n; ++age) {
        embed_add_byte(acc, emb, window_get(w, age), dim);
    }
    embed_acc_to_q8(out, acc, dim, n);
}