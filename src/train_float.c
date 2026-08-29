#include "train_float.h"
#include "data.h"
#include "embed.h"
#include "expert.h"
#include "expert_identity.h"
#include "expert_random.h"
#include "expert_train.h"
#include "head_f32.h"
#include "head_quant.h"
#include "head.h"
#include "loss.h"
#include "moe.h"
#include "context.h"
#include "window.h"
#include "ternary.h"
#include "sampling.h"
#include <stdio.h>

/*
 * ============================================================================
 * Configuration Constants
 * ============================================================================
 */

enum {
    VOCAB = 256,           /* Vocabulary size (byte-level) */
    DIM = 32,             /* Embedding/model dimension */
    HID = 64,             /* Hidden dimension for experts */
    E = 4,                /* Number of experts */
    CTX = 16,             /* Context length */
    DATA_MAX = 65536,     /* Maximum data size */
    RB1 = (DIM + 3) / 4,  /* Row bytes for W1 (ternary packing) */
    RB2 = (HID + 3) / 4,  /* Row bytes for W2 (ternary packing) */
    SCRATCH = (HID > DIM ? HID : DIM), /* Scratch buffer size */
    CKPT_MAGIC = 0x544D4F45u, /* Checkpoint magic number */
    CKPT_MAGIC_MOE = 0x544D4F32u /* MoE checkpoint magic */
};

/*
 * ============================================================================
 * Global State
 * ============================================================================
 */

static uint8_t *data = NULL;       /* Training data buffer */
static int data_len = 0;            /* Current data length */
static int data_capacity = 0;       /* Data buffer capacity */

/* Generation configuration */
static TrainFloatGenConfig gen_config;

/* Model weights */
static int8_t emb[VOCAB * DIM];                    /* Embedding matrix */
static uint8_t mw1[E * HID * RB1], mw2[E * DIM * RB2]; /* Expert weights */
static float head_w[VOCAB * DIM];                 /* Head weights (float32) */
static uint8_t head_q[VOCAB * RB1];                /* Quantized head (plane 0) */
static uint8_t head_q2[VOCAB * RB1];               /* Quantized head (plane 1) */
static float head_scale[VOCAB];                    /* Scale factors (plane 0) */
static float head_scale2[VOCAB];                  /* Scale factors (plane 1) */

/* Buffers */
static int8_t ctx_vec[DIM], moe_out[DIM], h[HID];
static int32_t ctx_acc[DIM], scratch[SCRATCH], logits_i[VOCAB];
static float logits_f[VOCAB], grad[VOCAB], scores[E];
static uint8_t wbuf[CTX];
static ByteWindow win;

/* Expert training state */
static int16_t expert_acc1[E * HID * DIM];
static int16_t expert_acc2[E * DIM * HID];

/* Context cache for fast training */
static int8_t ctx_cache[DATA_MAX * DIM];
static int fast_ready = 0;

/* Head backup for QAT */
static float head_backup[VOCAB * DIM];

/*
 * ============================================================================
 * Initialization
 * ============================================================================
 */

int train_float_load(const char *path) {
    /* Ensure we have enough capacity */
    if (data_capacity < DATA_MAX) {
        uint8_t *new_data = (uint8_t *)realloc(data, DATA_MAX);
        if (!new_data) {
            free(data);
            data = NULL;
            data_capacity = 0;
            data_len = 0;
            return 0;
        }
        data = new_data;
        data_capacity = DATA_MAX;
    }
    data_len = data_load_file(data, DATA_MAX, path);
    return data_len;
}

void train_float_init(void) {
    /* Initialize embeddings with hash pattern */
    embed_fill_hash(emb, VOCAB, DIM, 64);
    
    /* Initialize experts with identity */
    int wb1 = expert_w1_bytes(DIM, HID);
    int wb2 = expert_w2_bytes(DIM, HID);
    (void)wb1; (void)wb2;
    for (int e = 0; e < E; ++e) {
        expert_fill_identity(mw1 + e * wb1, mw2 + e * wb2, DIM, HID);
    }
    
    /* Initialize head weights to zero */
    head_f32_zero(head_w, VOCAB, DIM);
    
    /* Initialize scores */
    for (int i = 0; i < E; ++i) {
        scores[i] = (float)i;
    }
    
    /* Reset window */
    window_reset(&win, wbuf, CTX);
    
    /* Initialize generation config */
    gen_config = train_float_gen_config_default();
    
    /* Initialize data buffer */
    if (!data) {
        data = (uint8_t *)malloc(DATA_MAX);
        if (!data) {
            data_capacity = 0;
            data_len = 0;
            return;
        }
        data_capacity = DATA_MAX;
    }
}

/*
 * ============================================================================
 * Logits Conversion
 * ============================================================================
 */

static inline void logits_f32_to_i32(int32_t *out, const float *in, int n) {
    for (int i = 0; i < n; ++i) {
        float v = in[i];
        if (v > 1000000.0f) v = 1000000.0f;
        if (v < -1000000.0f) v = -1000000.0f;
        out[i] = (int32_t)v;
    }
}

static inline void logits_i32_to_f32(float *out, const int32_t *in, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = (float)in[i];
    }
}

/*
 * ============================================================================
 * Two-Plane Logits Computation
 * ============================================================================
 */

static void two_plane_logits_i32(int32_t *out, const int8_t *x) {
    for (int o = 0; o < VOCAB; ++o) {
        int a0 = tern_dot(x, head_q + o * RB1, DIM);
        int a1 = tern_dot(x, head_q2 + o * RB1, DIM);
        float v = (head_scale[o] * (float)a0 + head_scale2[o] * (float)a1) * 64.0f;
        if (v > 1000000.0f) v = 1000000.0f;
        if (v < -1000000.0f) v = -1000000.0f;
        out[o] = (int32_t)v;
    }
}

/*
 * ============================================================================
 * Training Epochs
 * ============================================================================
 */

float train_float_epoch(void) {
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0005f;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data_make_sample(data, data_len, pos, &win, CTX);
        if (target < 0 || target >= VOCAB) continue;
        
        context_vector_from_window(ctx_vec, ctx_acc, &win, emb, DIM, CTX);
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
        head_f32_logits(logits_f, head_w, moe_out, VOCAB, DIM);
        logits_f32_to_i32(logits_i, logits_f, VOCAB);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_head_mean_abs(void) {
    return head_f32_mean_abs(head_w, VOCAB * DIM);
}

float train_float_eval_quantized(float tau) {
    head_quantize_f32_to_ternary(head_q, head_w, VOCAB, DIM, tau);
    float total = 0.0f;
    int count = 0;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data_make_sample(data, data_len, pos, &win, CTX);
        if (target < 0 || target >= VOCAB) continue;
        
        context_vector_from_window(ctx_vec, ctx_acc, &win, emb, DIM, CTX);
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
        head_logits_i32(logits_i, head_q, moe_out, VOCAB, DIM);
        softmax_from_logits_f32(logits_f, logits_i, VOCAB);
        float loss = ce_loss_f32(logits_f, target, VOCAB);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_eval_quantized_scaled(void) {
    head_quantize_scaled(head_w, head_q, head_scale, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data_make_sample(data, data_len, pos, &win, CTX);
        if (target < 0 || target >= VOCAB) continue;
        
        context_vector_from_window(ctx_vec, ctx_acc, &win, emb, DIM, CTX);
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
        
        for (int o = 0; o < VOCAB; ++o) {
            int acc = tern_dot(moe_out, head_q + o * RB1, DIM);
            float v = head_scale[o] * (float)acc * 64.0f;
            if (v > 1000000.0f) v = 1000000.0f;
            if (v < -1000000.0f) v = -1000000.0f;
            logits_i[o] = (int32_t)v;
        }
        
        softmax_from_logits_f32(logits_f, logits_i, VOCAB);
        total += ce_loss_f32(logits_f, target, VOCAB);
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_eval_two_plane(void) {
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data_make_sample(data, data_len, pos, &win, CTX);
        if (target < 0 || target >= VOCAB) continue;
        
        context_vector_from_window(ctx_vec, ctx_acc, &win, emb, DIM, CTX);
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
        two_plane_logits_i32(logits_i, moe_out);
        softmax_from_logits_f32(logits_f, logits_i, VOCAB);
        total += ce_loss_f32(logits_f, target, VOCAB);
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_qat_epoch(void) {
    head_quantize_scaled(head_w, head_q, head_scale, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0002f;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data_make_sample(data, data_len, pos, &win, CTX);
        if (target < 0 || target >= VOCAB) continue;
        
        context_vector_from_window(ctx_vec, ctx_acc, &win, emb, DIM, CTX);
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
        
        for (int o = 0; o < VOCAB; ++o) {
            int acc = tern_dot(moe_out, head_q + o * RB1, DIM);
            float v = head_scale[o] * (float)acc * 64.0f;
            if (v > 1000000.0f) v = 1000000.0f;
            if (v < -1000000.0f) v = -1000000.0f;
            logits_i[o] = (int32_t)v;
        }
        
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_qat_two_plane_epoch(void) {
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0001f;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
        int target = data_make_sample(data, data_len, pos, &win, CTX);
        if (target < 0 || target >= VOCAB) continue;
        
        context_vector_from_window(ctx_vec, ctx_acc, &win, emb, DIM, CTX);
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
        two_plane_logits_i32(logits_i, moe_out);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

void train_float_prepare_two_plane(void) {
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
}

int train_float_copy_tail(uint8_t *out, int max_len) {
    if (data_len <= 0 || max_len <= 0) return 0;
    int n = data_len < max_len ? data_len : max_len;
    int start = data_len - n;
    for (int i = 0; i < n; ++i) out[i] = data[start + i];
    return n;
}

int train_float_next_two_plane(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
    two_plane_logits_i32(logits_i, moe_out);
    
    /* Find best token */
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}

/*
 * ============================================================================
 * Checkpoint Save/Load
 * ============================================================================
 */

int train_float_save_checkpoint(const char *path) {
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    uint32_t magic = CKPT_MAGIC;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    
    fwrite(&magic, 4, 1, f);
    fwrite(head_w, 1, sizeof(head_w), f);
    fwrite(head_q, 1, sizeof(head_q), f);
    fwrite(head_q2, 1, sizeof(head_q2), f);
    fwrite(head_scale, 1, sizeof(head_scale), f);
    fwrite(head_scale2, 1, sizeof(head_scale2), f);
    
    fclose(f);
    return 0;
}

int train_float_load_checkpoint(const char *path) {
    uint32_t magic = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    
    if (fread(&magic, 4, 1, f) != 1 || magic != CKPT_MAGIC) {
        fclose(f);
        return -1;
    }
    
    if (fread(head_w, 1, sizeof(head_w), f) != sizeof(head_w)) { fclose(f); return -1; }
    if (fread(head_q, 1, sizeof(head_q), f) != sizeof(head_q)) { fclose(f); return -1; }
    if (fread(head_q2, 1, sizeof(head_q2), f) != sizeof(head_q2)) { fclose(f); return -1; }
    if (fread(head_scale, 1, sizeof(head_scale), f) != sizeof(head_scale)) { fclose(f); return -1; }
    if (fread(head_scale2, 1, sizeof(head_scale2), f) != sizeof(head_scale2)) { fclose(f); return -1; }
    
    fclose(f);
    return 0;
}

/*
 * ============================================================================
 * Fast Training (Context Cache)
 * ============================================================================
 */

static inline void fast_relu(int8_t *y, const int8_t *x, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

static void build_context_cache(void) {
    uint8_t buf[CTX];
    ByteWindow w;
    window_reset(&w, buf, CTX);
    
    for (int pos = 0; pos < data_len; ++pos) {
        context_vector_from_window(ctx_cache + pos * DIM, ctx_acc, &w, emb, DIM, CTX);
        window_push(&w, data[pos]);
    }
    
    fast_ready = 1;
}

void train_float_fast_prepare(void) {
    build_context_cache();
}

float train_float_epoch_fast(void) {
    if (!fast_ready) build_context_cache();
    
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0005f;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        fast_relu(moe_out, ctx, DIM);
        head_f32_logits(logits_f, head_w, moe_out, VOCAB, DIM);
        logits_f32_to_i32(logits_i, logits_f, VOCAB);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_qat_two_plane_epoch_fast(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0001f;
    const int interval = 64;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        if (count > 0 && (count % interval) == 0) {
            head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
        }
        
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        fast_relu(moe_out, ctx, DIM);
        two_plane_logits_i32(logits_i, moe_out);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_eval_two_plane_fast(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        fast_relu(moe_out, ctx, DIM);
        two_plane_logits_i32(logits_i, moe_out);
        softmax_from_logits_f32(logits_f, logits_i, VOCAB);
        total += ce_loss_f32(logits_f, target, VOCAB);
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_finetune_fast(int normal, int qat) {
    float loss = 0.0f;
    for (int e = 0; e < normal; ++e) loss = train_float_epoch_fast();
    for (int e = 0; e < qat; ++e) loss = train_float_qat_two_plane_epoch_fast();
    return loss;
}

int train_float_next_two_plane_fast(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    fast_relu(moe_out, ctx_vec, DIM);
    two_plane_logits_i32(logits_i, moe_out);
    
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}

/*
 * ============================================================================
 * QAT Fast V2
 * ============================================================================
 */

void train_float_backup_head(void) {
    for (int i = 0; i < VOCAB * DIM; ++i) {
        head_backup[i] = head_w[i];
    }
}

void train_float_restore_head(void) {
    for (int i = 0; i < VOCAB * DIM; ++i) {
        head_w[i] = head_backup[i];
    }
}

float train_float_qat_two_plane_epoch_fast2(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0001f;
    
    int samples = data_len > CTX ? data_len - CTX : 0;
    int interval = samples <= 64 ? 1 : (samples <= 512 ? 8 : 64);
    
    for (int pos = CTX; pos < data_len; ++pos) {
        if (count > 0 && (count % interval) == 0) {
            head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
        }
        
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        fast_relu(moe_out, ctx, DIM);
        two_plane_logits_i32(logits_i, moe_out);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

/*
 * ============================================================================
 * MoE Training (Random Experts)
 * ============================================================================
 */

static inline void route_by_prev(uint8_t b) {
    int e = (int)(b % E);
    for (int i = 0; i < E; ++i) {
        scores[i] = (i == e) ? 1.0f : 0.0f;
    }
}

void train_float_init_random_experts(void) {
    int wb1 = expert_w1_bytes(DIM, HID);
    int wb2 = expert_w2_bytes(DIM, HID);
    (void)wb1; (void)wb2;
    for (int e = 0; e < E; ++e) {
        expert_fill_random(mw1 + e * wb1, mw2 + e * wb2, DIM, HID, e * 7919 + 17);
    }
}

float train_float_epoch_fast_moe(void) {
    if (!fast_ready) build_context_cache();
    
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0005f;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        route_by_prev(data[pos - 1]);
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        head_f32_logits(logits_f, head_w, moe_out, VOCAB, DIM);
        logits_f32_to_i32(logits_i, logits_f, VOCAB);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_qat_two_plane_epoch_fast_moe(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0001f;
    
    int samples = data_len > CTX ? data_len - CTX : 0;
    int interval = samples <= 64 ? 1 : (samples <= 512 ? 8 : 64);
    
    for (int pos = CTX; pos < data_len; ++pos) {
        if (count > 0 && (count % interval) == 0) {
            head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
        }
        
        route_by_prev(data[pos - 1]);
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        two_plane_logits_i32(logits_i, moe_out);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_eval_two_plane_fast_moe(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    
    for (int pos = CTX; pos < data_len; ++pos) {
        route_by_prev(data[pos - 1]);
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        two_plane_logits_i32(logits_i, moe_out);
        softmax_from_logits_f32(logits_f, logits_i, VOCAB);
        total += ce_loss_f32(logits_f, target, VOCAB);
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

int train_float_next_two_plane_fast_moe(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    route_by_prev(ctx_len > 0 ? ctx[ctx_len - 1] : 0);
    moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
    two_plane_logits_i32(logits_i, moe_out);
    
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}

/*
 * ============================================================================
 * MoE Checkpoint
 * ============================================================================
 */

int train_float_save_checkpoint_moe(const char *path) {
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    uint32_t magic = CKPT_MAGIC_MOE;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    
    fwrite(&magic, 4, 1, f);
    fwrite(head_w, 1, sizeof(head_w), f);
    fwrite(head_q, 1, sizeof(head_q), f);
    fwrite(head_q2, 1, sizeof(head_q2), f);
    fwrite(head_scale, 1, sizeof(head_scale), f);
    fwrite(head_scale2, 1, sizeof(head_scale2), f);
    fwrite(mw1, 1, sizeof(mw1), f);
    fwrite(mw2, 1, sizeof(mw2), f);
    
    fclose(f);
    return 0;
}

int train_float_load_checkpoint_moe(const char *path) {
    uint32_t magic = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    
    if (fread(&magic, 4, 1, f) != 1 || magic != CKPT_MAGIC_MOE) {
        fclose(f);
        return -1;
    }
    
    if (fread(head_w, 1, sizeof(head_w), f) != sizeof(head_w)) { fclose(f); return -1; }
    if (fread(head_q, 1, sizeof(head_q), f) != sizeof(head_q)) { fclose(f); return -1; }
    if (fread(head_q2, 1, sizeof(head_q2), f) != sizeof(head_q2)) { fclose(f); return -1; }
    if (fread(head_scale, 1, sizeof(head_scale), f) != sizeof(head_scale)) { fclose(f); return -1; }
    if (fread(head_scale2, 1, sizeof(head_scale2), f) != sizeof(head_scale2)) { fclose(f); return -1; }
    if (fread(mw1, 1, sizeof(mw1), f) != sizeof(mw1)) { fclose(f); return -1; }
    if (fread(mw2, 1, sizeof(mw2), f) != sizeof(mw2)) { fclose(f); return -1; }
    
    fclose(f);
    return 0;
}

/*
 * ============================================================================
 * Expert Training
 * ============================================================================
 */

void train_float_zero_expert_acc(void) {
    for (int i = 0; i < E * HID * DIM; ++i) expert_acc1[i] = 0;
    for (int i = 0; i < E * DIM * HID; ++i) expert_acc2[i] = 0;
}

static inline void head_f32_grad_x_fast(float *gx, const float *grad) {
    for (int i = 0; i < DIM; ++i) gx[i] = 0.0f;
    for (int o = 0; o < VOCAB; ++o) {
        float g = grad[o];
        if (g == 0.0f) continue;
        const float *row = head_w + o * DIM;
        for (int i = 0; i < DIM; ++i) {
            gx[i] += g * row[i];
        }
    }
}

float train_float_epoch_fast_moe_trainable(int expert_thr) {
    if (!fast_ready) build_context_cache();
    
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0005f;
    static float gx[DIM];
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int e = (int)(data[pos - 1] % E);
        route_by_prev(data[pos - 1]);
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        head_f32_logits(logits_f, head_w, moe_out, VOCAB, DIM);
        logits_f32_to_i32(logits_i, logits_f, VOCAB);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_grad_x_fast(gx, grad);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        int wb1 = expert_w1_bytes(DIM, HID);
        int wb2 = expert_w2_bytes(DIM, HID);
        (void)wb1; (void)wb2;
        uint8_t *w1 = mw1 + e * wb1;
        uint8_t *w2 = mw2 + e * wb2;
        int16_t *a1 = expert_acc1 + e * HID * DIM;
        int16_t *a2 = expert_acc2 + e * DIM * HID;
        
        expert_update_w1(w1, a1, w2, gx, h, ctx, DIM, HID, expert_thr);
        expert_update_w2(w2, a2, gx, h, DIM, HID, expert_thr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

/*
 * ============================================================================
 * Expert0 Training (Single Expert)
 * ============================================================================
 */

static inline void route_first(void) {
    for (int i = 0; i < E; ++i) scores[i] = (i == 0) ? 1.0f : 0.0f;
}

float train_float_epoch_fast_expert0_trainable(int expert_thr) {
    if (!fast_ready) build_context_cache();
    
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0005f;
    static float gx[DIM];
    
    route_first();
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        head_f32_logits(logits_f, head_w, moe_out, VOCAB, DIM);
        logits_f32_to_i32(logits_i, logits_f, VOCAB);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_grad_x_fast(gx, grad);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        int wb1 = expert_w1_bytes(DIM, HID);
        int wb2 = expert_w2_bytes(DIM, HID);
        (void)wb1; (void)wb2;
        uint8_t *w1 = mw1;
        uint8_t *w2 = mw2;
        int16_t *a1 = expert_acc1;
        int16_t *a2 = expert_acc2;
        
        expert_update_w1(w1, a1, w2, gx, h, ctx, DIM, HID, expert_thr);
        expert_update_w2(w2, a2, gx, h, DIM, HID, expert_thr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_qat_two_plane_epoch_fast_expert0(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    const float lr = 0.0001f;
    
    int samples = data_len > CTX ? data_len - CTX : 0;
    int interval = samples <= 64 ? 1 : (samples <= 512 ? 8 : 64);
    
    route_first();
    
    for (int pos = CTX; pos < data_len; ++pos) {
        if (count > 0 && (count % interval) == 0) {
            head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
        }
        
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        two_plane_logits_i32(logits_i, moe_out);
        float loss = softmax_ce_loss_and_grad_f32(grad, logits_i, target, VOCAB);
        head_f32_train_step(head_w, grad, moe_out, VOCAB, DIM, lr);
        
        total += loss;
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

float train_float_eval_two_plane_fast_expert0(void) {
    if (!fast_ready) build_context_cache();
    
    head_quantize_two_plane(head_w, head_q, head_q2, head_scale, head_scale2, VOCAB, DIM);
    float total = 0.0f;
    int count = 0;
    
    route_first();
    
    for (int pos = CTX; pos < data_len; ++pos) {
        int target = data[pos];
        const int8_t *ctx = ctx_cache + pos * DIM;
        
        moe_forward_top1(moe_out, mw1, mw2, scores, ctx, E, DIM, HID, 0, 0, h, scratch);
        two_plane_logits_i32(logits_i, moe_out);
        softmax_from_logits_f32(logits_f, logits_i, VOCAB);
        total += ce_loss_f32(logits_f, target, VOCAB);
        count++;
    }
    
    return count > 0 ? total / (float)count : 0.0f;
}

int train_float_next_two_plane_fast_expert0(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    route_first();
    moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
    two_plane_logits_i32(logits_i, moe_out);
    
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}

/*
 * ============================================================================
 * Generation with Sampling Support
 * ============================================================================
 */

void train_float_set_gen_config(const TrainFloatGenConfig *cfg) {
    gen_config = *cfg;
    sampling_set_seed(cfg->seed);
}

void train_float_get_gen_config(TrainFloatGenConfig *cfg) {
    *cfg = gen_config;
}

/*
 * Generation with sampling for MoE model.
 */

int train_float_next_two_plane_fast_moe_with_sampling(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    route_by_prev(ctx_len > 0 ? ctx[ctx_len - 1] : 0);
    moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
    two_plane_logits_i32(logits_i, moe_out);
    
    /* Use sampling if configured */
    if (gen_config.temperature > 0.001f && gen_config.top_k > 0) {
        logits_i32_to_f32(logits_f, logits_i, VOCAB);
        return sample_top_k(logits_f, VOCAB, gen_config.top_k, gen_config.temperature);
    } else if (gen_config.temperature > 0.001f) {
        logits_i32_to_f32(logits_f, logits_i, VOCAB);
        return sample_with_temperature(logits_f, VOCAB, gen_config.temperature);
    }
    
    /* Fall back to argmax */
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}

/*
 * Generation with sampling for expert0 model.
 */

int train_float_next_two_plane_fast_expert0_with_sampling(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    route_first();
    moe_forward_top1(moe_out, mw1, mw2, scores, ctx_vec, E, DIM, HID, 0, 0, h, scratch);
    two_plane_logits_i32(logits_i, moe_out);
    
    /* Use sampling if configured */
    if (gen_config.temperature > 0.001f && gen_config.top_k > 0) {
        logits_i32_to_f32(logits_f, logits_i, VOCAB);
        return sample_top_k(logits_f, VOCAB, gen_config.top_k, gen_config.temperature);
    } else if (gen_config.temperature > 0.001f) {
        logits_i32_to_f32(logits_f, logits_i, VOCAB);
        return sample_with_temperature(logits_f, VOCAB, gen_config.temperature);
    }
    
    /* Fall back to argmax */
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}

/*
 * Generation with sampling for regular (non-MoE) model.
 */

int train_float_next_two_plane_fast_with_sampling(const uint8_t *ctx, int ctx_len) {
    uint8_t buf[CTX];
    ByteWindow wloc;
    window_reset(&wloc, buf, CTX);
    int start = ctx_len > CTX ? ctx_len - CTX : 0;
    for (int i = start; i < ctx_len; ++i) window_push(&wloc, ctx[i]);
    context_vector_from_window(ctx_vec, ctx_acc, &wloc, emb, DIM, CTX);
    fast_relu(moe_out, ctx_vec, DIM);
    two_plane_logits_i32(logits_i, moe_out);
    
    /* Use sampling if configured */
    if (gen_config.temperature > 0.001f && gen_config.top_k > 0) {
        logits_i32_to_f32(logits_f, logits_i, VOCAB);
        return sample_top_k(logits_f, VOCAB, gen_config.top_k, gen_config.temperature);
    } else if (gen_config.temperature > 0.001f) {
        logits_i32_to_f32(logits_f, logits_i, VOCAB);
        return sample_with_temperature(logits_f, VOCAB, gen_config.temperature);
    }
    
    /* Fall back to argmax */
    int best = 0;
    int32_t best_v = logits_i[0];
    for (int o = 1; o < VOCAB; ++o) {
        if (logits_i[o] > best_v) {
            best_v = logits_i[o];
            best = o;
        }
    }
    
    return best;
}
