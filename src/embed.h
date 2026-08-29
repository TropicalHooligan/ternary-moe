#ifndef EMBED_H
#define EMBED_H

#include <stdint.h>

void embed_fill_pattern(int8_t *emb, int vocab, int dim);
void embed_fill_hash(int8_t *emb, int vocab, int dim, int scale);
const int8_t* embed_row(const int8_t *emb, int token, int dim);
void embed_add_byte(int32_t *acc, const int8_t *emb, int token, int dim);
void embed_acc_to_q8(int8_t *out, const int32_t *acc, int dim, int count);

#endif