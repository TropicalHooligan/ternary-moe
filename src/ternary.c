#include "ternary.h"

/*
 * Ternary weights are packed 4-per-byte, 2 bits each:
 *   00 -> 0   01 -> +1   10 -> -1   11 -> 0 (unused, kept for symmetry)
 *
 * TERN_LUT stays exported since it's part of the public ABI (tests and
 * older callers may reference it), but the hot paths below decode a
 * 2-bit code with plain arithmetic instead of a table lookup:
 *
 *   code   bit1 bit0   (bit0) - (bit1)
 *   00        0    0        0
 *   01        0    1        1
 *   10        1    0       -1
 *   11        1    1        0
 *
 * That's one AND, one shift+AND and one SUB -- no memory indirection --
 * which matters because tern_dot's inner loop runs millions of times
 * per training epoch.
 */
const int8_t TERN_LUT[4] = {0, 1, -1, 0};

static inline int32_t tern_code_to_val(unsigned code) {
    return (int32_t)(code & 1u) - (int32_t)((code >> 1) & 1u);
}

uint8_t tern_pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    uint8_t code = 0;
    code |= (a > 0 ? 1 : (a < 0 ? 2 : 0));
    code |= (b > 0 ? 1 : (b < 0 ? 2 : 0)) << 2;
    code |= (c > 0 ? 1 : (c < 0 ? 2 : 0)) << 4;
    code |= (d > 0 ? 1 : (d < 0 ? 2 : 0)) << 6;
    return code;
}

int8_t tern_get(const uint8_t *p, int i) {
    unsigned code = (p[i >> 2] >> (2 * (i & 3))) & 3u;
    return (int8_t)tern_code_to_val(code);
}

int8_t tern_quant(float w, float tau) {
    if (w > tau) return 1;
    if (w < -tau) return -1;
    return 0;
}

/*
 * n is always a multiple of 4 in every real caller (DIM=32, HID=64 and
 * their row-byte-counts (n+3)/4 are exact), so we walk whole packed
 * bytes -- decoding all 4 lanes with shifts instead of recomputing
 * i>>2 / i&3 on every element -- and only fall back to the generic
 * per-element path for a possible short tail.
 */
int32_t tern_dot(const int8_t *restrict x, const uint8_t *restrict w, int n) {
    int32_t s = 0;
    int i = 0;
    int full_bytes = n >> 2;

    for (int b = 0; b < full_bytes; ++b) {
        unsigned byte = w[b];
        s += x[i]     * tern_code_to_val(byte & 3u);
        s += x[i + 1] * tern_code_to_val((byte >> 2) & 3u);
        s += x[i + 2] * tern_code_to_val((byte >> 4) & 3u);
        s += x[i + 3] * tern_code_to_val((byte >> 6) & 3u);
        i += 4;
    }
    for (; i < n; ++i) {
        s += (int32_t)x[i] * (int32_t)tern_get(w, i);
    }
    return s;
}

void tern_set(uint8_t *p, int i, int8_t v) {
    uint8_t code = v > 0 ? 1 : (v < 0 ? 2 : 0);
    int shift = 2 * (i & 3);
    uint8_t mask = (uint8_t)(3 << shift);
    p[i >> 2] = (uint8_t)((p[i >> 2] & ~mask) | (code << shift));
}

void tern_adjust(uint8_t *p, int i, int dir) {
    int8_t v = tern_get(p, i);
    if (dir > 0 && v < 1) v++;
    if (dir < 0 && v > -1) v--;
    tern_set(p, i, v);
}
