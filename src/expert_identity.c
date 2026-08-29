#include "expert_identity.h"
#include "ternary.h"

void expert_fill_identity(uint8_t *w1, uint8_t *w2, int dim, int hidden) {
    int rb1 = (dim + 3) / 4;
    int rb2 = (hidden + 3) / 4;
    int n1 = hidden * rb1;
    int n2 = dim * rb2;
    for (int i = 0; i < n1; ++i) w1[i] = 0;
    for (int i = 0; i < n2; ++i) w2[i] = 0;
    int m1 = hidden < dim ? hidden : dim;
    for (int o = 0; o < m1; ++o) tern_set(w1 + o * rb1, o, 1);
    int m2 = dim < hidden ? dim : hidden;
    for (int o = 0; o < m2; ++o) tern_set(w2 + o * rb2, o, 1);
}