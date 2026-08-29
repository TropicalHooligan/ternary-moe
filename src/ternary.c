#include "ternary.h"

/*
 * Ternary weights are packed 4-per-byte, 2 bits each:
 *   00 -> 0   01 -> +1   10 -> -1   11 -> 0 (unused, kept for symmetry)
 *
 * TERN_LUT stays exported since it's part of the public ABI (tests and
 * older callers may reference it), but the hot paths below decode a
 * 2-bit code with plain arithmetic instead of a table lookup.
 */
const int8_t TERN_LUT[4] = {0, 1, -1, 0};
