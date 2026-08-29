#include "sampling.h"
#include <stdio.h>

int main(void) {
    /* Test random number generation */
    sampling_set_seed(42);
    float r1 = sampling_random_float();
    float r2 = sampling_random_float();
    if (r1 < 0.0f || r1 >= 1.0f) return 1;
    if (r2 < 0.0f || r2 >= 1.0f) return 1;
    
    /* Test random int */
    sampling_set_seed(123);
    int n = 100;
    int rint = sampling_random_int(n);
    if (rint < 0 || rint >= n) return 1;
    
    /* Test argmax with temperature = 0 */
    float logits[4] = {1.0f, 5.0f, 2.0f, 3.0f};
    int result = sample_with_temperature(logits, 4, 0.0f);
    if (result != 1) return 1; /* Should pick index 1 (value 5.0) */
    
    /* Test that temperature > 0 gives different results */
    sampling_set_seed(42);
    result = sample_with_temperature(logits, 4, 1.0f);
    /* With temperature, we should get a valid index */
    if (result < 0 || result >= 4) return 1;
    
    /* Test top-k sampling */
    sampling_set_seed(42);
    result = sample_top_k(logits, 4, 2, 1.0f);
    if (result < 0 || result >= 4) return 1;
    /* With top-k=2, should only sample from indices 1 and 3 (values 5.0 and 3.0) */
    /* But we can't guarantee which one, just that it's valid */
    
    /* Test softmax */
    float probs[4];
    softmax_with_temperature(probs, logits, 4, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (probs[i] <= 0.0f) return 1; /* All probabilities should be positive */
        sum += probs[i];
    }
    if (sum < 0.99f || sum > 1.01f) return 1; /* Sum should be ~1.0 */
    
    /* Test find_top_k */
    int indices[2];
    find_top_k(indices, logits, 4, 2);
    /* Should return indices 1 and 3 (values 5.0 and 3.0) in descending order */
    /* Note: our implementation returns sorted indices (highest first) */
    if (indices[0] != 1) return 1;
    if (indices[1] != 3) return 1;
    
    printf("PASS\n");
    return 0;
}
