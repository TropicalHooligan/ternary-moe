#include <stdio.h>
#include <stddef.h>
#include "window.h"

static int test_window_basic(void) {
    uint8_t buf[4];
    ByteWindow w;
    window_reset(&w, buf, 4);

    window_push(&w, 10);
    window_push(&w, 20);
    window_push(&w, 30);

    return window_len(&w) == 3 &&
           window_get(&w, 0) == 30 &&
           window_get(&w, 1) == 20 &&
           window_get(&w, 2) == 10;
}

static int test_window_wrap(void) {
    uint8_t buf[3];
    ByteWindow w;
    window_reset(&w, buf, 3);

    window_push(&w, 1);
    window_push(&w, 2);
    window_push(&w, 3);
    window_push(&w, 4);

    return window_len(&w) == 3 &&
           window_get(&w, 0) == 4 &&
           window_get(&w, 1) == 3 &&
           window_get(&w, 2) == 2;
}

static int test_window_out_of_range(void) {
    uint8_t buf[2];
    ByteWindow w;
    window_reset(&w, buf, 2);

    window_push(&w, 7);

    return window_get(&w, -1) == 0 &&
           window_get(&w, 1) == 0 &&
           window_get(&w, 100) == 0;
}

static int test_window_zero_cap(void) {
    ByteWindow w;
    window_reset(&w, NULL, 0);

    window_push(&w, 1);

    return window_len(&w) == 0 &&
           window_get(&w, 0) == 0;
}

int main(void) {
    int ok = 1;
    ok &= test_window_basic();
    ok &= test_window_wrap();
    ok &= test_window_out_of_range();
    ok &= test_window_zero_cap();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}