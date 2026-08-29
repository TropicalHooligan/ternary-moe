#include <stdio.h>
#include "train_float.h"

static int test_checkpoint_moe(void) {
    const char *data_path = "build/tmp_moe_ckpt_data.txt";
    const char *model_path = "build/tmp_moe_ckpt.tmo";
    FILE *f = fopen(data_path, "wb");
    if (!f) return 0;
    for (int i = 0; i < 128; ++i) fputc('a', f);
    fclose(f);
    if (train_float_load(data_path) <= 16) return 0;
    train_float_init();
    train_float_init_random_experts();
    train_float_fast_prepare();
    train_float_epoch_fast_moe();
    if (train_float_save_checkpoint_moe(model_path) != 0) return 0;
    uint8_t ctx[16];
    int len = train_float_copy_tail(ctx, 16);
    int b1 = train_float_next_two_plane_fast_moe(ctx, len);
    train_float_init();
    if (train_float_load_checkpoint_moe(model_path) != 0) return 0;
    int b2 = train_float_next_two_plane_fast_moe(ctx, len);
    return b1 == b2;
}

int main(void) {
    int ok = 1;
    ok &= test_checkpoint_moe();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}