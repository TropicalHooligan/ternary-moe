#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "train_float.h"
#include "common/gen_ctx.h"

enum { GEN_CTX = 16, DEFAULT_GEN_LEN = 200 };

static void print_usage(const char *prog) {
    printf("usage: %s <model> <seed_file> [gen_len] [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --temperature, -t <val>   Temperature for sampling (default: 0.8)\n");
    printf("  --top-k, -k <n>           Top-k sampling (default: 50)\n");
    printf("  --top-p, -p <val>         Top-p/nucleus sampling (default: 0.0, disabled)\n");
    printf("  --seed, -s <n>            Random seed (default: 12345)\n");
    printf("  --argmax                 Use pure argmax (greedy) decoding\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
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

    int gen_len = DEFAULT_GEN_LEN;
    TrainFloatGenConfig cfg = train_float_gen_config_default();
    
    /* Parse optional arguments */
    int i = 3;
    while (i < argc) {
        if (strcmp(argv[i], "--temperature") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                cfg.temperature = (float)atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "-k") == 0) {
            if (i + 1 < argc) {
                cfg.top_k = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--top-p") == 0 || strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                cfg.top_p = (float)atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) {
                cfg.seed = (uint32_t)atol(argv[++i]);
            }
        } else if (strcmp(argv[i], "--argmax") == 0) {
            cfg.temperature = 0.0f;  // Disable sampling, use pure argmax
            cfg.top_k = 0;
        } else if (i == 3) {
            /* First positional argument after model and seed is gen_len */
            gen_len = atoi(argv[i]);
        }
        i++;
    }

    /* If gen_len wasn't set from positional arg, check if it's numeric */
    if (gen_len == DEFAULT_GEN_LEN && argc > 3) {
        char *endptr;
        int val = (int)strtol(argv[3], &endptr, 10);
        if (endptr != argv[3] && val > 0) {
            gen_len = val;
        }
    }

    if (gen_len <= 0) gen_len = DEFAULT_GEN_LEN;

    /* Set generation config */
    train_float_set_gen_config(&cfg);

    uint8_t ctx[GEN_CTX];
    int ctx_len = train_float_copy_tail(ctx, GEN_CTX);

    for (int i = 0; i < gen_len; ++i) {
        int b = train_float_next_two_plane_fast_moe_with_sampling(ctx, ctx_len);
        putchar(gen_ctx_printable(b));
        gen_ctx_append(ctx, GEN_CTX, &ctx_len, (uint8_t)b);
    }
    putchar('\n');
    return 0;
}
