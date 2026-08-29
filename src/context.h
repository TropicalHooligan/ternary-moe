#ifndef CONTEXT_H
#define CONTEXT_H

#include <stdint.h>
#include "window.h"

void context_vector_from_window(int8_t *out, int32_t *acc, const ByteWindow *w, const int8_t *emb, int dim, int ctx);

#endif