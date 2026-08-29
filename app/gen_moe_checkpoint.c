#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "train_float.h"
#include "common/gen_ctx.h"

enum { GEN_CTX = 16, DEFAULT_GEN_LEN = 200 };

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: gen_moe_checkpoint <model> <seed_file> [gen_len]\n");
        return 1;
    }

    train_float_init();
    if (train_float_load_checkpoint_moe(argv[1]) != 0) {
        printf("load model failed\n");
        return 1;
    }
    if (train_float_load(argv[2]) <= 0) {
        printf("load seed failed\n");
        return 1;
    }

    int gen_len = argc > 3 ? atoi(argv[3]) : DEFAULT_GEN_LEN;
    if (gen_len <= 0) gen_len = 1;

    uint8_t ctx[GEN_CTX];
    int ctx_len = train_float_copy_tail(ctx, GEN_CTX);

    for (int i = 0; i < gen_len; ++i) {
        int b = train_float_next_two_plane_fast_moe(ctx, ctx_len);
        putchar(gen_ctx_printable(b));
        gen_ctx_append(ctx, GEN_CTX, &ctx_len, (uint8_t)b);
    }
    putchar('\n');
    return 0;
}
