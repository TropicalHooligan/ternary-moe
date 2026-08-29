#include "sampling.h"
#include "loss.h"
#include <math.h>
#include <string.h>

/* Simple xorshift PRNG for reproducible sampling */
static uint32_t rng_state = 123456789;

void sampling_set_seed(uint32_t seed) {
    if (seed == 0) seed = 1; // Avoid zero seed
    rng_state = seed;
}

static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

float sampling_random_float(void) {
    /* Generate float in [0, 1) using 23 bits of precision */
    return (float)(xorshift32() >> 8) / (float)(0x7FFFFFFUL);
}

int sampling_random_int(int n) {
    if (n <= 0) return 0;
    /* Rejection sampling to avoid modulo bias */
    uint32_t max = 0x7FFFFFFF - (0x7FFFFFFF % (uint32_t)n);
    uint32_t x;
    do {
        x = xorshift32();
    } while (x >= max);
    return (int)(x % (uint32_t)n);
}

/*
 * Find the index of the maximum value in an array.
 * Used as fallback when temperature = 0.
 */
static int argmax(const float *arr, int n) {
    int best = 0;
    float best_val = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    return best;
}

/*
 * Compute softmax with temperature and numerical stability.
 * 
 * The temperature is applied before softmax:
 *   logits_i = logits[i] / temperature
 *   probs[i] = exp(logits_i - max_logits) / sum(exp(logits_j - max_logits))
 */
void softmax_with_temperature(float *probs, const float *logits, int vocab, float temperature) {
    float max_logit = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    float sum = 0.0f;
    float inv_temp = (temperature > 0.001f) ? (1.0f / temperature) : 1e9f;
    
    for (int i = 0; i < vocab; ++i) {
        float scaled = (logits[i] - max_logit) * inv_temp;
        /* Clamp to prevent overflow */
        if (scaled > 50.0f) scaled = 50.0f;
        if (scaled < -50.0f) scaled = -50.0f;
        probs[i] = expf(scaled);
        sum += probs[i];
    }
    
    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab; ++i) {
        probs[i] *= inv_sum;
    }
}

/*
 * Sample from a probability distribution using inverse transform sampling.
 */
static int sample_from_probs(const float *probs, int vocab) {
    float r = sampling_random_float();
    float cumulative = 0.0f;
    
    for (int i = 0; i < vocab; ++i) {
        cumulative += probs[i];
        if (cumulative >= r) {
            return i;
        }
    }
    
    /* Fallback to last index (shouldn't happen with proper normalization) */
    return vocab - 1;
}

/*
 * Sample with temperature.
 * When temperature is 0, falls back to argmax.
 */
int sample_with_temperature(const float *logits, int vocab, float temperature) {
    if (temperature <= 0.001f || temperature > 100.0f) {
        /* Very low or very high temperature: use argmax */
        return argmax(logits, vocab);
    }
    
    /* Compute softmax with temperature */
    float *probs = (float *)malloc(vocab * sizeof(float));
    if (!probs) {
        return argmax(logits, vocab);
    }
    
    softmax_with_temperature(probs, logits, vocab, temperature);
    int result = sample_from_probs(probs, vocab);
    free(probs);
    
    return result;
}

/*
 * Simple partition-based top-k selection.
 * Not the most efficient but works well for small vocabularies (256).
 */
void find_top_k(int *indices, const float *logits, int vocab, int k) {
    if (k >= vocab) {
        for (int i = 0; i < vocab; ++i) {
            indices[i] = i;
        }
        return;
    }
    
    /* Initialize with first k indices */
    for (int i = 0; i < k; ++i) {
        indices[i] = i;
    }
    
    /* Sort the first k elements */
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (logits[indices[j]] > logits[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }
    
    /* For remaining elements, check if they belong in top-k */
    for (int i = k; i < vocab; ++i) {
        /* Find minimum in current top-k */
        int min_idx = 0;
        for (int j = 1; j < k; ++j) {
            if (logits[indices[j]] < logits[indices[min_idx]]) {
                min_idx = j;
            }
        }
        
        if (logits[i] > logits[indices[min_idx]]) {
            indices[min_idx] = i;
            /* Re-sort this position */
            for (int j = min_idx + 1; j < k; ++j) {
                if (logits[indices[j]] > logits[indices[min_idx]]) {
                    int tmp = indices[min_idx];
                    indices[min_idx] = indices[j];
                    indices[j] = tmp;
                    min_idx = j;
                }
            }
        }
    }
}

/*
 * Sample from top-k candidates.
 */
int sample_top_k(const float *logits, int vocab, int k, float temperature) {
    if (k <= 0) k = 1;
    if (k >= vocab) {
        /* Top-k equals full vocabulary, just use temperature sampling */
        return sample_with_temperature(logits, vocab, temperature);
    }
    
    /* Find top-k indices */
    int *top_indices = (int *)malloc(k * sizeof(int));
    if (!top_indices) {
        return argmax(logits, vocab);
    }
    
    find_top_k(top_indices, logits, vocab, k);
    
    /* Extract top-k logits */
    float *top_logits = (float *)malloc(k * sizeof(float));
    if (!top_logits) {
        free(top_indices);
        return argmax(logits, vocab);
    }
    
    for (int i = 0; i < k; ++i) {
        top_logits[i] = logits[top_indices[i]];
    }
    
    /* Sample from top-k with temperature */
    int result_idx = sample_with_temperature(top_logits, k, temperature);
    int result = top_indices[result_idx];
    
    free(top_logits);
    free(top_indices);
    
    return result;
}

/*
 * Nucleus (top-p) sampling.
 * Selects the smallest set of tokens whose cumulative probability >= p.
 */
int sample_nucleus(const float *logits, int vocab, float p, float temperature) {
    if (p <= 0.0f) p = 0.9f;
    if (p >= 1.0f) {
        /* p=1 means use full vocabulary */
        return sample_with_temperature(logits, vocab, temperature);
    }
    
    /* Compute softmax with temperature */
    float *probs = (float *)malloc(vocab * sizeof(float));
    if (!probs) {
        return argmax(logits, vocab);
    }
    
    softmax_with_temperature(probs, logits, vocab, temperature);
    
    /* Find nucleus set */
    int nucleus_size = 0;
    float cumulative = 0.0f;
    
    /* Create index array sorted by probability */
    int *sorted_indices = (int *)malloc(vocab * sizeof(int));
    if (!sorted_indices) {
        free(probs);
        return argmax(logits, vocab);
    }
    
    for (int i = 0; i < vocab; ++i) {
        sorted_indices[i] = i;
    }
    
    /* Simple bubble sort (vocab is only 256) */
    for (int i = 0; i < vocab - 1; ++i) {
        for (int j = i + 1; j < vocab; ++j) {
            if (probs[sorted_indices[j]] > probs[sorted_indices[i]]) {
                int tmp = sorted_indices[i];
                sorted_indices[i] = sorted_indices[j];
                sorted_indices[j] = tmp;
            }
        }
    }
    
    /* Find how many tokens we need */
    for (nucleus_size = 0; nucleus_size < vocab; ++nucleus_size) {
        cumulative += probs[sorted_indices[nucleus_size]];
        if (cumulative >= p) {
            nucleus_size++;
            break;
        }
    }
    
    /* Sample from nucleus */
    float r = sampling_random_float() * cumulative;
    cumulative = 0.0f;
    int result = sorted_indices[0];
    
    for (int i = 0; i < nucleus_size; ++i) {
        cumulative += probs[sorted_indices[i]];
        if (cumulative >= r) {
            result = sorted_indices[i];
            break;
        }
    }
    
    free(sorted_indices);
    free(probs);
    
    return result;
}
