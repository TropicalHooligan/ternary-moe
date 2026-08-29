#ifndef SAMPLING_H
#define SAMPLING_H

#include <stdint.h>
#include <stdlib.h>

/*
 * Temperature sampling and top-k sampling for improved generation quality.
 * 
 * These functions provide alternatives to pure argmax (greedy) decoding,
 * which tends to produce repetitive output, especially with under-trained models.
 */

/*
 * Sample from a probability distribution with temperature.
 * 
 * Args:
 *   logits - array of logits (unnormalized log probabilities)
 *   vocab - size of vocabulary (256 for byte-level)
 *   temperature - controls randomness:
 *                 - temperature = 0: equivalent to argmax (greedy)
 *                 - temperature < 1: sharper distribution (more deterministic)
 *                 - temperature = 1: original softmax distribution
 *                 - temperature > 1: flatter distribution (more random)
 * 
 * Returns: sampled index
 */
int sample_with_temperature(const float *logits, int vocab, float temperature);

/*
 * Sample from top-k candidates with temperature.
 * 
 * First selects the k highest-scoring tokens, then samples from them
 * using temperature softmax. This prevents the model from generating
 * very low-probability (often nonsensical) tokens.
 * 
 * Args:
 *   logits - array of logits
 *   vocab - size of vocabulary
 *   k - number of top candidates to consider
 *   temperature - temperature for sampling among top-k
 * 
 * Returns: sampled index from top-k
 */
int sample_top_k(const float *logits, int vocab, int k, float temperature);

/*
 * Nucleus (top-p) sampling.
 * 
 * Selects the smallest set of top tokens whose cumulative probability
 * exceeds p, then samples from that set.
 * 
 * Args:
 *   logits - array of logits
 *   vocab - size of vocabulary
 *   p - probability threshold (0.0-1.0)
 *   temperature - temperature for sampling
 * 
 * Returns: sampled index
 */
int sample_nucleus(const float *logits, int vocab, float p, float temperature);

/*
 * Softmax with overflow protection.
 * 
 * Computes softmax probabilities from logits, handling numerical stability.
 * 
 * Args:
 *   probs - output array for probabilities (must be pre-allocated, size vocab)
 *   logits - input logits
 *   vocab - size of vocabulary
 */
void softmax_with_temperature(float *probs, const float *logits, int vocab, float temperature);

/*
 * Find top-k indices (for top-k sampling).
 * 
 * Returns the indices of the k largest values in logits.
 * Uses a simple partial sort algorithm.
 * 
 * Args:
 *   indices - output array for top-k indices (must be pre-allocated, size k)
 *   logits - input logits
 *   vocab - size of vocabulary
 *   k - number of top indices to find
 */
void find_top_k(int *indices, const float *logits, int vocab, int k);

/*
 * Random number generation utilities.
 * Uses a simple xorshift PRNG for reproducibility and speed.
 */

/* Initialize random seed */
void sampling_set_seed(uint32_t seed);

/* Get random float in [0, 1) */
float sampling_random_float(void);

/* Get random int in [0, n) */
int sampling_random_int(int n);

#endif // SAMPLING_H
