#include <stdio.h>
#include <stdlib.h>
#include "train_float.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: train_save_moe <data> <model> [normal] [qat]\n");
        return 1;
    }

    int n = train_float_load(argv[1]);
    if (n <= 16) {
        printf("data too small: %d\n", n);
        return 1;
    }

    int normal = argc > 3 ? atoi(argv[3]) : 3;
    int qat = argc > 4 ? atoi(argv[4]) : 3;
    if (normal <= 0) normal = 1;
    if (qat <= 0) qat = 1;

    train_float_init();
    train_float_init_random_experts();
    train_float_fast_prepare();

    for (int e = 0; e < normal; ++e) {
        float loss = train_float_epoch_fast_moe();
        printf("moe normal %d loss %.4f\n", e, loss);
    }

    float eval_normal = train_float_eval_two_plane_fast_moe();
    printf("moe normal eval %.4f\n", eval_normal);
    train_float_backup_head();

    for (int e = 0; e < qat; ++e) {
        float loss = train_float_qat_two_plane_epoch_fast_moe();
        printf("moe qat %d loss %.4f\n", e, loss);
    }

    float eval_qat = train_float_eval_two_plane_fast_moe();
    printf("moe qat eval %.4f\n", eval_qat);
    if (eval_qat > eval_normal + 0.2f) {
        train_float_restore_head();
        eval_qat = eval_normal;
        printf("restored head\n");
    }
    printf("final moe eval %.4f\n", eval_qat < eval_normal ? eval_qat : eval_normal);

    if (train_float_save_checkpoint_moe(argv[2]) != 0) {
        printf("save failed\n");
        return 1;
    }
    printf("saved %s\n", argv[2]);
    return 0;
}
