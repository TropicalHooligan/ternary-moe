#include "window.h"

void window_reset(ByteWindow *w, uint8_t *buf, int cap) {
    w->buf = buf;
    w->cap = cap > 0 ? cap : 0;
    w->len = 0;
    w->pos = 0;
}

void window_push(ByteWindow *w, uint8_t b) {
    if (w->cap <= 0) return;
    w->buf[w->pos] = b;
    w->pos++;
    if (w->pos >= w->cap) w->pos = 0;
    if (w->len < w->cap) w->len++;
}

int window_len(const ByteWindow *w) {
    return w->len;
}

uint8_t window_get(const ByteWindow *w, int age) {
    if (age < 0 || age >= w->len || w->cap <= 0) return 0;
    int idx = w->pos - 1 - age;
    if (idx < 0) idx += w->cap;
    return w->buf[idx];
}