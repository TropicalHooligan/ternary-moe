#include <stdio.h>
#include "context.h"
#include "embed.h"
#include "window.h"

static int8_t emb[256 * 4];

static int test_context_empty(void) {
    uint8_t buf[4];
    ByteWindow w;
    window_reset(&w, buf, 4);

    int8_t out[4];
    int32_t acc[4];

    context_vector_from_window(out, acc, &w, emb, 4, 4);

    return out[0] == 0 &&
           out[1] == 0 &&
           out[2] == 0 &&
           out[3] == 0;
}

static int test_context_one(void) {
    uint8_t buf[4];
    ByteWindow w;
    window_reset(&w, buf, 4);

    window_push(&w, 1);

    int8_t out[4];
    int32_t acc[4];

    context_vector_from_window(out, acc, &w, emb, 4, 4);

    return out[0] == 16 &&
           out[1] == -16 &&
           out[2] == 16 &&
           out[3] == -16;
}

static int test_context_two_avg(void) {
    uint8_t buf[4];
    ByteWindow w;
    window_reset(&w, buf, 4);

    window_push(&w, 1);
    window_push(&w, 2);

    int8_t out[4];
    int32_t acc[4];

    context_vector_from_window(out, acc, &w, emb, 4, 4);

    return out[0] == 0 &&
           out[1] == 0 &&
           out[2] == 0 &&
           out[3] == 0;
}

static int test_context_limit(void) {
    uint8_t buf[4];
    ByteWindow w;
    window_reset(&w, buf, 4);

    window_push(&w, 1);
    window_push(&w, 2);

    int8_t out[4];
    int32_t acc[4];

    context_vector_from_window(out, acc, &w, emb, 4, 1);

    return out[0] == -16 &&
           out[1] == 16 &&
           out[2] == -16 &&
           out[3] == 16;
}

int main(void) {
    embed_fill_pattern(emb, 256, 4);

    int ok = 1;
    ok &= test_context_empty();
    ok &= test_context_one();
    ok &= test_context_two_avg();
    ok &= test_context_limit();

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}