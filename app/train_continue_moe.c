#include <stdio.h>
#include <stdlib.h>
#include "train_float.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: train_continue_moe <model> <data> [normal] [qat]\n");
        return 1;
    }

    train_float_init();
    if (train_float_load_checkpoint_moe(argv[1]) != 0) {
        printf("load model failed\n");
        return 1;
    }

    int n = train_float_load(argv[2]);
    if (n <= 16) {
        printf("data too small: %d\n", n);
        return 1;
    }

    int normal = argc > 3 ? atoi(argv[3]) : 1;
    int qat = argc > 4 ? atoi(argv[4]) : 1;
    if (normal <= 0) normal = 0;
    if (qat <= 0) qat = 0;

    train_float_fast_prepare();
    for (int e = 0; e < normal; ++e) {
        float loss = train_float_epoch_fast_moe();
        printf("moe normal %d loss %.4f\n", e, loss);
    }

    float mid = train_float_eval_two_plane_fast_moe();
    printf("mid moe eval %.4f\n", mid);
    train_float_backup_head();

    for (int e = 0; e < qat; ++e) {
        float loss = train_float_qat_two_plane_epoch_fast_moe();
        printf("moe qat %d loss %.4f\n", e, loss);
    }

    float after = train_float_eval_two_plane_fast_moe();
    printf("after moe eval %.4f\n", after);
    if (after > mid + 0.2f) {
        train_float_restore_head();
        after = mid;
        printf("restored head\n");
    }
    printf("final moe eval %.4f\n", after);

    if (train_float_save_checkpoint_moe(argv[1]) != 0) {
        printf("save failed\n");
        return 1;
    }
    printf("saved %s\n", argv[1]);
    return 0;
}
