#include <stdio.h>
#include "loss.h"

static int near(float a, float b, float eps) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    return d <= eps;
}

static int test_softmax_uniform(void) {
    int32_t logits[3] = {0, 0, 0};
    float p[3];

    softmax_from_logits_f32(p, logits, 3);

    float sum = p[0] + p[1] + p[2];

    return near(p[0], 1.0f / 3.0f, 1e-4f) &&
           near(p[1], 1.0f / 3.0f, 1e-4f) &&
           near(p[2], 1.0f / 3.0f, 1e-4f) &&
           near(sum, 1.0f, 1e-4f);
}

static int test_ce_loss_uniform(void) {
    float p[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};

    float loss = ce_loss_f32(p, 1, 3);

    return near(loss, 1.098612f, 1e-3f);
}

static int test_grad_uniform(void) {
    float p[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
    float grad[3];

    softmax_ce_grad_f32(grad, p, 1, 3);

    return near(grad[0], 1.0f / 3.0f, 1e-4f) &&
           near(grad[1], -2.0f / 3.0f, 1e-4f) &&
           near(grad[2], 1.0f / 3.0f, 1e-4f);
}

static int test_combined_stable(void) {
    int32_t logits[3] = {0, 1000, 0};
    float grad[3];

    float loss = softmax_ce_loss_and_grad_f32(grad, logits, 1, 3);

    return loss < 1e-3f &&
           grad[0] > -1e-3f && grad[0] < 1e-3f &&
           grad[1] > -1e-3f && grad[1] < 1e-3f &&
           grad[2] > -1e-3f && grad[2] < 1e-3f;
}

static int test_combined_zero_size(void) {
    return softmax_ce_loss_and_grad_f32(0, 0, -1, 0) == 0.0f;
}

int main(void) {
    int ok = 1;
    ok &= test_softmax_uniform();
    ok &= test_ce_loss_uniform();
    ok &= test_grad_uniform();
    ok &= test_combined_stable();
    ok &= test_combined_zero_size();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}