#include <stdio.h>
#include "embed.h"

static int test_fill_pattern(void) {
    int8_t emb[8];
    embed_fill_pattern(emb, 2, 4);

    return emb[0] == -16 &&
           emb[1] == 16 &&
           emb[2] == -16 &&
           emb[3] == 16 &&
           emb[4] == 16 &&
           emb[5] == -16;
}

static int test_row(void) {
    int8_t emb[8];
    embed_fill_pattern(emb, 2, 4);

    const int8_t *r0 = embed_row(emb, 0, 4);
    const int8_t *r1 = embed_row(emb, 1, 4);

    return r0 == emb && r1 == emb + 4;
}

static int test_add_byte(void) {
    int8_t emb[8];
    embed_fill_pattern(emb, 2, 4);

    int32_t acc[4] = {0, 0, 0, 0};
    embed_add_byte(acc, emb, 1, 4);

    return acc[0] == 16 &&
           acc[1] == -16 &&
           acc[2] == 16 &&
           acc[3] == -16;
}

static int test_acc_to_q8_clamp(void) {
    int32_t acc[3] = {300, -300, 20};
    int8_t out[3];

    embed_acc_to_q8(out, acc, 3, 1);

    return out[0] == 127 &&
           out[1] == -127 &&
           out[2] == 20;
}

static int test_acc_to_q8_avg(void) {
    int32_t acc[2] = {100, -100};
    int8_t out[2];

    embed_acc_to_q8(out, acc, 2, 2);

    return out[0] == 50 && out[1] == -50;
}

static int test_acc_to_q8_zero_count(void) {
    int32_t acc[1] = {99};
    int8_t out[1];

    embed_acc_to_q8(out, acc, 1, 0);

    return out[0] == 0;
}

int main(void) {
    int ok = 1;
    ok &= test_fill_pattern();
    ok &= test_row();
    ok &= test_add_byte();
    ok &= test_acc_to_q8_clamp();
    ok &= test_acc_to_q8_avg();
    ok &= test_acc_to_q8_zero_count();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}