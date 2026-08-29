#include "data.h"
#include <stdio.h>

int data_load_file(uint8_t *buf, int max_len, const char *path) {
    if (max_len <= 0) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t r = fread(buf, 1, (size_t)max_len, f);
    fclose(f);
    return (int)r;
}

int data_make_sample(const uint8_t *data, int len, int pos, ByteWindow *w, int ctx) {
    if (!data || !w || pos < 0 || pos >= len) return -1;
    window_reset(w, w->buf, w->cap);
    int start = pos - ctx;
    if (start < 0) start = 0;
    for (int i = start; i < pos; ++i) {
        window_push(w, data[i]);
    }
    return data[pos];
}