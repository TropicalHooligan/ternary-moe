#ifndef DATA_H
#define DATA_H

#include <stdint.h>
#include "window.h"

int data_load_file(uint8_t *buf, int max_len, const char *path);
int data_make_sample(const uint8_t *data, int len, int pos, ByteWindow *w, int ctx);

#endif