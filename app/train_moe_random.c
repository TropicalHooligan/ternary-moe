#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "train_float.h"
#include "common/gen_ctx.h"

enum { GEN_CTX = 16 };

static void generate_moe(int gen_len) {
    uint8_t ctx[GEN_CTX];
    int ctx_len = train_float_copy_tail(ctx, GEN_CTX);

    for (int i = 0; i < gen_len; ++i) {
        int b = train_float_next_two_plane_fast_moe(ctx, ctx_len);
        putchar(gen_ctx_printable(b));
        gen_ctx_append(ctx, GEN_CTX, &ctx_len, (uint8_t)b);
    }
    putchar('\n');
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: train_moe_random <file> [normal] [qat] [gen_len]\n");
        return 1;
    }

    int n = train_float_load(argv[1]);
    if (n <= 16) {
        printf("data too small: %d\n", n);
        return 1;
    }

    int normal = argc > 2 ? atoi(argv[2]) : 2;
    int qat = argc > 3 ? atoi(argv[3]) : 2;
    int gen = argc > 4 ? atoi(argv[4]) : 80;
    if (normal <= 0) normal = 1;
    if (qat <= 0) qat = 1;
    if (gen <= 0) gen = 1;

    train_float_init();
    train_float_init_random_experts();
    train_float_fast_prepare();

    for (int e = 0; e < normal; ++e) {
        float loss = train_float_epoch_fast_moe();
        printf("moe normal %d loss %.4f\n", e, loss);
    }
    printf("moe normal eval %.4f\n", train_float_eval_two_plane_fast_moe());

    for (int e = 0; e < qat; ++e) {
        float loss = train_float_qat_two_plane_epoch_fast_moe();
        printf("moe qat %d loss %.4f\n", e, loss);
    }
    printf("moe qat eval %.4f\n", train_float_eval_two_plane_fast_moe());

    train_float_prepare_two_plane();
    generate_moe(gen);
    return 0;
}
