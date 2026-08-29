#include <stdio.h>
#include <stddef.h>
#include "router.h"

static int test_top1(void) {
    float s[4] = {0.1f, 0.9f, 0.2f, 0.5f};
    return router_top1(s, 4) == 1;
}

static int test_top1_empty(void) {
    return router_top1(NULL, 0) == -1;
}

static int test_top2(void) {
    float s[4] = {0.1f, 0.9f, 0.8f, 0.5f};
    int a = -1;
    int b = -1;
    router_top2(s, 4, &a, &b);
    return a == 1 && b == 2;
}

static int test_top2_empty(void) {
    int a = 0;
    int b = 0;
    router_top2(NULL, 0, &a, &b);
    return a == -1 && b == -1;
}

static int test_top2_single(void) {
    float s[1] = {2.0f};
    int a = -1;
    int b = -1;
    router_top2(s, 1, &a, &b);
    return a == 0 && b == 0;
}

static int test_top2_tie(void) {
    float s[3] = {1.0f, 1.0f, 0.0f};
    int a = -1;
    int b = -1;
    router_top2(s, 3, &a, &b);
    return a == 0 && b == 1;
}

int main(void) {
    int ok = 1;
    ok &= test_top1();
    ok &= test_top1_empty();
    ok &= test_top2();
    ok &= test_top2_empty();
    ok &= test_top2_single();
    ok &= test_top2_tie();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}