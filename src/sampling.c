#include "sampling.h"
#include "loss.h"
#include <math.h>
#include <string.h>
#include <limits.h>

/*
 * ============================================================================
 * Random Number Generator
 * ============================================================================
 * 
 * Simple xorshift PRNG for reproducible sampling.
 * Uses a 32-bit state with good statistical properties.
 * 
 * Period: 2^32 - 1
 * Passes: BigCrush (with proper seeding)
 */

static uint32_t rng_state = 123456789;

void sampling_set_seed(uint32_t seed) {
    /* Avoid zero seed - it causes all zeros to be generated */
    rng_state = seed ? seed : 1;
}

/*
 * Xorshift32 - Fast PRNG with good statistical properties.
 * ~3x faster than LCG, passes statistical tests.
 */
static inline uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

/*
 * Generate random float in [0, 1) with 23 bits of precision.
 * Uses 24 bits from the PRNG (discarding the lowest 8 bits for better quality).
 */
float sampling_random_float(void) {
    /* Use upper 24 bits for better distribution quality */
    return (float)(xorshift32() >> 8) * (1.0f / 16777216.0f);
}

/*
 * Generate random int in [0, n) using rejection sampling.
 * Avoids modulo bias by rejecting values in the upper range.
 */
int sampling_random_int(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 0;
    
    /* Rejection sampling to avoid modulo bias */
    /* Find largest multiple of n that fits in 2^31-1 */
    uint32_t max = UINT32_MAX - (UINT32_MAX % (uint32_t)n);
    uint32_t x;
    
    do {
        x = xorshift32();
    } while (x >= max);
    
    return (int)(x % (uint32_t)n);
}

/*
 * ============================================================================
 * Softmax with Temperature
 * ============================================================================
 * 
 * Numerical stability: subtract max before exp
 * Overflow protection: clamp scaled values
 * 
 * Optimizations:
 * - Single pass for max finding
 * - Fused exp and accumulation
 * - Precomputed inverse temperature
 */

void softmax_with_temperature(float *probs, const float *logits, int vocab, float temperature) {
    /* Find max for numerical stability */
    float max_logit = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    /* Compute inverse temperature (handle edge cases) */
    float inv_temp = temperature;
    if (temperature < 1e-6f) {
        /* Very low temperature: approximate with argmax behavior */
        inv_temp = 1e9f;
    } else if (temperature > 100.0f) {
        /* Very high temperature: flatten distribution */
        inv_temp = 1e-2f;
    } else {
        inv_temp = 1.0f / temperature;
    }
    
    /* Compute exp(logit - max) * inv_temp and accumulate */
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        float scaled = (logits[i] - max_logit) * inv_temp;
        
        /* Clamp to prevent overflow/underflow */
        /* exp(50) ~ 5e21, exp(-50) ~ 2e-22 - safe range for float */
        if (scaled > 50.0f) scaled = 50.0f;
        else if (scaled < -50.0f) scaled = -50.0f;
        
        probs[i] = expf(scaled);
        sum += probs[i];
    }
    
    /* Normalize - use reciprocal for speed */
    float inv_sum = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < vocab; ++i) {
        probs[i] *= inv_sum;
    }
}

/*
 * ============================================================================
 * Argmax - Find Maximum
 * ============================================================================
 * 
 * Simple linear scan for maximum value.
 * Used as fallback when temperature = 0 or for deterministic sampling.
 */

static inline int argmax(const float *arr, int n) {
    int best = 0;
    float best_val = arr[0];
    
    /* Unroll loop for better performance on small vocabularies */
    int i = 1;
    for (; i + 3 < n; i += 4) {
        if (arr[i] > best_val) { best = i; best_val = arr[i]; }
        if (arr[i+1] > best_val) { best = i+1; best_val = arr[i+1]; }
        if (arr[i+2] > best_val) { best = i+2; best_val = arr[i+2]; }
        if (arr[i+3] > best_val) { best = i+3; best_val = arr[i+3]; }
    }
    
    /* Handle remaining elements */
    for (; i < n; ++i) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    
    return best;
}

/*
 * ============================================================================
 * Sampling from Probability Distribution
 * ============================================================================
 * 
 * Inverse Transform Sampling:
 * 1. Generate random number r in [0, 1)
 * 2. Find first index where cumulative probability >= r
 * 
 * Optimized for small vocabularies (256 tokens).
 */

static inline int sample_from_probs(const float *probs, int vocab) {
    float r = sampling_random_float();
    float cumulative = 0.0f;
    
    /* Unroll loop for small vocabularies */
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
 * ============================================================================
 * Sample with Temperature
 * ============================================================================
 * 
 * Main sampling function with temperature control.
 * 
 * Args:
 *   logits - unnormalized log probabilities
 *   vocab - size of vocabulary
 *   temperature - controls randomness (0 = argmax, 1 = softmax, >1 = more random)
 * 
 * Returns: sampled index
 */

int sample_with_temperature(const float *logits, int vocab, float temperature) {
    /* Edge case: very low or very high temperature */
    if (temperature <= 0.001f) {
        return argmax(logits, vocab);
    }
    
    if (temperature > 100.0f) {
        /* Very high temperature: nearly uniform distribution */
        return sampling_random_int(vocab);
    }
    
    /* For small vocabularies, allocate on stack instead of heap */
    if (vocab <= 256) {
        float probs[256];
        softmax_with_temperature(probs, logits, vocab, temperature);
        return sample_from_probs(probs, vocab);
    } else {
        /* Fallback to heap allocation for larger vocabularies */
        float *probs = (float *)malloc(vocab * sizeof(float));
        if (!probs) {
            return argmax(logits, vocab);
        }
        softmax_with_temperature(probs, logits, vocab, temperature);
        int result = sample_from_probs(probs, vocab);
        free(probs);
        return result;
    }
}

/*
 * ============================================================================
 * Top-K Selection (Heap-based)
 * ============================================================================
 * 
 * Efficient top-k selection using a min-heap.
 * 
 * For small vocabularies (<= 256) and small k (<= 64), we use a simple
 * selection algorithm that's cache-friendly.
 * 
 * Complexity: O(n log k) vs O(n log n) for full sort
 * 
 * Optimizations:
 * - Use a fixed-size array on stack for small k
 * - Minimize memory allocations
 * - Cache-friendly access patterns
 */

/*
 * Simple selection-based top-k for small vocabularies.
 * For vocab=256 and k<=64, this is very efficient.
 */
void find_top_k(int *indices, const float *logits, int vocab, int k) {
    if (k <= 0) k = 1;
    if (k >= vocab) {
        /* Return all indices */
        for (int i = 0; i < vocab; ++i) {
            indices[i] = i;
        }
        return;
    }
    
    /* Use stack allocation for small k */
    if (k <= 64) {
        /* Initialize with first k indices */
        float values[64];
        for (int i = 0; i < k; ++i) {
            indices[i] = i;
            values[i] = logits[i];
        }
        
        /* Sort initial k in descending order */
        for (int i = 1; i < k; ++i) {
            float val = values[i];
            int idx = indices[i];
            int j = i;
            while (j > 0 && values[j-1] < val) {
                values[j] = values[j-1];
                indices[j] = indices[j-1];
                j--;
            }
            values[j] = val;
            indices[j] = idx;
        }
        
        /* Process remaining elements */
        for (int i = k; i < vocab; ++i) {
            float val = logits[i];
            
            /* Check if this value belongs in top-k */
            if (val > values[k-1]) {
                /* Find insertion point to maintain descending order */
                int j = k - 1;
                while (j > 0 && values[j-1] < val) {
                    values[j] = values[j-1];
                    indices[j] = indices[j-1];
                    j--;
                }
                values[j] = val;
                indices[j] = i;
            }
        }
    } else {
        /* Fallback to simple partial sort for larger k */
        /* Initialize with first k indices */
        float *values = (float *)malloc(k * sizeof(float));
        if (!values) {
            /* Fallback to first k */
            for (int i = 0; i < k; ++i) indices[i] = i;
            return;
        }
        
        for (int i = 0; i < k; ++i) {
            indices[i] = i;
            values[i] = logits[i];
        }
        
        /* Simple insertion sort for initial k */
        for (int i = 1; i < k; ++i) {
            float val = values[i];
            int idx = indices[i];
            int j = i;
            while (j > 0 && values[j-1] < val) {
                values[j] = values[j-1];
                indices[j] = indices[j-1];
                j--;
            }
            values[j] = val;
            indices[j] = idx;
        }
        
        /* Process remaining elements */
        for (int i = k; i < vocab; ++i) {
            float val = logits[i];
            if (val > values[0]) {
                int j = 0;
                while (j < k - 1 && values[j+1] < val) {
                    values[j] = values[j+1];
                    indices[j] = indices[j+1];
                    j++;
                }
                values[j] = val;
                indices[j] = i;
            }
        }
        
        free(values);
    }
}

/*
 * ============================================================================
 * Sample from Top-K Candidates
 * ============================================================================
 * 
 * Two-stage sampling:
 * 1. Find top-k indices
 * 2. Sample from those k using temperature softmax
 * 
 * This prevents the model from generating very low-probability tokens.
 */

int sample_top_k(const float *logits, int vocab, int k, float temperature) {
    if (k <= 0) k = 1;
    
    /* If k >= vocab, just sample from full vocabulary */
    if (k >= vocab) {
        return sample_with_temperature(logits, vocab, temperature);
    }
    
    /* Find top-k indices */
    if (vocab <= 256 && k <= 64) {
        /* Use stack allocation for small vocabularies */
        int top_indices[64];
        find_top_k(top_indices, logits, vocab, k);
        
        /* Extract top-k logits */
        float top_logits[64];
        for (int i = 0; i < k; ++i) {
            top_logits[i] = logits[top_indices[i]];
        }
        
        /* Sample from top-k with temperature */
        float probs[64];
        softmax_with_temperature(probs, top_logits, k, temperature);
        int result_idx = sample_from_probs(probs, k);
        
        return top_indices[result_idx];
    } else {
        /* Fallback to heap allocation */
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
}

/*
 * ============================================================================
 * Nucleus (Top-p) Sampling
 * ============================================================================
 * 
 * Selects the smallest set of tokens whose cumulative probability >= p.
 * Then samples uniformly from that set.
 * 
 * This is also known as "top-p sampling" or "nucleus sampling".
 * 
 * Optimizations:
 * - Single pass through sorted probabilities
 * - Early exit when cumulative probability exceeds p
 * - Stack allocation for small vocabularies
 */

int sample_nucleus(const float *logits, int vocab, float p, float temperature) {
    /* Clamp p to valid range */
    if (p <= 0.0f) p = 0.01f;
    if (p >= 1.0f) {
        /* p=1 means use full vocabulary */
        return sample_with_temperature(logits, vocab, temperature);
    }
    
    /* For small vocabularies, use stack allocation */
    if (vocab <= 256) {
        float probs[256];
        int sorted_indices[256];
        
        /* Compute softmax with temperature */
        softmax_with_temperature(probs, logits, vocab, temperature);
        
        /* Create sorted indices using simple sort (vocab is small) */
        for (int i = 0; i < vocab; ++i) {
            sorted_indices[i] = i;
        }
        
        /* Simple insertion sort for small arrays */
        for (int i = 1; i < vocab; ++i) {
            float val = probs[i];
            int idx = sorted_indices[i];
            int j = i;
            while (j > 0 && probs[sorted_indices[j-1]] < val) {
                sorted_indices[j] = sorted_indices[j-1];
                j--;
            }
            sorted_indices[j] = idx;
        }
        
        /* Find nucleus set */
        float cumulative = 0.0f;
        int nucleus_size = 0;
        
        for (; nucleus_size < vocab; ++nucleus_size) {
            cumulative += probs[sorted_indices[nucleus_size]];
            if (cumulative >= p) {
                nucleus_size++;
                break;
            }
        }
        
        /* Sample from nucleus */
        if (nucleus_size <= 0) {
            /* Fallback: return first index if nucleus is empty */
            nucleus_size = 1;
            cumulative = probs[sorted_indices[0]];
            if (cumulative <= 0.0f) cumulative = 1.0f; /* Avoid division by zero */
        }
        
        float r = sampling_random_float() * cumulative;
        cumulative = 0.0f;
        
        for (int i = 0; i < nucleus_size; ++i) {
            cumulative += probs[sorted_indices[i]];
            if (cumulative >= r) {
                return sorted_indices[i];
            }
        }
        
        return sorted_indices[nucleus_size - 1];
    } else {
        /* Fallback to heap allocation for larger vocabularies */
        float *probs = (float *)malloc(vocab * sizeof(float));
        if (!probs) {
            return argmax(logits, vocab);
        }
        
        softmax_with_temperature(probs, logits, vocab, temperature);
        
        int *sorted_indices = (int *)malloc(vocab * sizeof(int));
        if (!sorted_indices) {
            free(probs);
            return argmax(logits, vocab);
        }
        
        for (int i = 0; i < vocab; ++i) {
            sorted_indices[i] = i;
        }
        
        /* Simple insertion sort */
        for (int i = 1; i < vocab; ++i) {
            float val = probs[i];
            int idx = sorted_indices[i];
            int j = i;
            while (j > 0 && probs[sorted_indices[j-1]] < val) {
                sorted_indices[j] = sorted_indices[j-1];
                j--;
            }
            sorted_indices[j] = idx;
        }
        
        float cumulative = 0.0f;
        int nucleus_size = 0;
        
        for (; nucleus_size < vocab; ++nucleus_size) {
            cumulative += probs[sorted_indices[nucleus_size]];
            if (cumulative >= p) {
                nucleus_size++;
                break;
            }
        }
        
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
}
