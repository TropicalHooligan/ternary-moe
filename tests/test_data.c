#include <stdio.h>
#include "data.h"
#include "window.h"

static int test_data_make_sample(void) {
    uint8_t dbuf[6] = {97, 98, 99, 100, 101, 102};
    uint8_t wbuf[4];
    ByteWindow w;
    window_reset(&w, wbuf, 4);
    int target = data_make_sample(dbuf, 6, 3, &w, 3);
    return target == 100 &&
           window_get(&w, 0) == 99 &&
           window_get(&w, 1) == 98 &&
           window_get(&w, 2) == 97;
}

static int test_data_load_file(void) {
    const char *path = "build/tmp_data.bin";
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fputc(1, f);
    fputc(2, f);
    fclose(f);
    uint8_t buf[4];
    int n = data_load_file(buf, 4, path);
    return n == 2 && buf[0] == 1 && buf[1] == 2;
}

int main(void) {
    int ok = 1;
    ok &= test_data_make_sample();
    ok &= test_data_load_file();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}