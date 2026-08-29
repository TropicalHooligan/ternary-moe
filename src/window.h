#ifndef WINDOW_H
#define WINDOW_H

#include <stdint.h>

typedef struct {
    uint8_t *buf;
    int cap;
    int len;
    int pos;
} ByteWindow;

void window_reset(ByteWindow *w, uint8_t *buf, int cap);
void window_push(ByteWindow *w, uint8_t b);
int window_len(const ByteWindow *w);
uint8_t window_get(const ByteWindow *w, int age);

#endif