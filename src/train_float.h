#ifndef TRAIN_FLOAT_H
#define TRAIN_FLOAT_H

#include <stdint.h>

/* Generation configuration */
typedef struct {
    float temperature;
    int top_k;
    float top_p;
    uint32_t seed;
} TrainFloatGenConfig;

/* Default generation config */
static inline TrainFloatGenConfig train_float_gen_config_default(void) {
    TrainFloatGenConfig cfg;
    cfg.temperature = 0.8f;
    cfg.top_k = 50;
    cfg.top_p = 0.0f;
    cfg.seed = 12345;
    return cfg;
}

int train_float_load(const char *path);
void train_float_init(void);

float train_float_epoch(void);
float train_float_qat_epoch(void);
float train_float_qat_two_plane_epoch(void);

float train_float_eval_quantized(float tau);
float train_float_eval_quantized_scaled(void);
float train_float_eval_two_plane(void);

float train_float_head_mean_abs(void);

void train_float_prepare_two_plane(void);
int train_float_next_two_plane(const uint8_t *ctx, int ctx_len);
int train_float_copy_tail(uint8_t *out, int max_len);

int train_float_save_checkpoint(const char *path);
int train_float_load_checkpoint(const char *path);

void train_float_fast_prepare(void);
float train_float_epoch_fast(void);
float train_float_qat_two_plane_epoch_fast(void);
float train_float_eval_two_plane_fast(void);
int train_float_next_two_plane_fast(const uint8_t *ctx, int ctx_len);

float train_float_qat_two_plane_epoch_fast2(void);
void train_float_backup_head(void);
void train_float_restore_head(void);

void train_float_init_random_experts(void);
float train_float_epoch_fast_moe(void);
float train_float_qat_two_plane_epoch_fast_moe(void);
float train_float_eval_two_plane_fast_moe(void);
int train_float_next_two_plane_fast_moe(const uint8_t *ctx, int ctx_len);

int train_float_save_checkpoint_moe(const char *path);
int train_float_load_checkpoint_moe(const char *path);

void train_float_zero_expert_acc(void);
float train_float_epoch_fast_moe_trainable(int expert_thr);

float train_float_epoch_fast_expert0_trainable(int expert_thr);
float train_float_qat_two_plane_epoch_fast_expert0(void);
float train_float_eval_two_plane_fast_expert0(void);
int train_float_next_two_plane_fast_expert0(const uint8_t *ctx, int ctx_len);

/* New functions with sampling support */
void train_float_set_gen_config(const TrainFloatGenConfig *cfg);
void train_float_get_gen_config(TrainFloatGenConfig *cfg);
int train_float_next_two_plane_fast_moe_with_sampling(const uint8_t *ctx, int ctx_len);
int train_float_next_two_plane_fast_expert0_with_sampling(const uint8_t *ctx, int ctx_len);
int train_float_next_two_plane_fast_with_sampling(const uint8_t *ctx, int ctx_len);

#endif