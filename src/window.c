#include "window.h"

/*
 * ============================================================================
 * Sliding Window Implementation
 * ============================================================================
 * 
 * A circular buffer for maintaining a sliding window of tokens.
 * Used for efficient context management.
 */

/*
 * Reset window to empty state.
 * 
 * Args:
 *   w - window to reset
 *   buf - buffer for token storage
 *   cap - capacity (maximum number of tokens)
 */

void window_reset(ByteWindow *w, uint8_t *buf, int cap) {
    w->buf = buf;
    w->cap = cap > 0 ? cap : 0;
    w->len = 0;
    w->pos = 0;
}

/*
 * Push a token into the window.
 * 
 * Args:
 *   w - window
 *   b - token to push
 * 
 * The window maintains the last 'cap' tokens in a circular buffer.
 */

void window_push(ByteWindow *w, uint8_t b) {
    if (w->cap <= 0) return;
    
    /* Store token at current position */
    w->buf[w->pos] = b;
    
    /* Advance position (wrap around if needed) */
    w->pos++;
    if (w->pos >= w->cap) w->pos = 0;
    
    /* Increase length (up to capacity) */
    if (w->len < w->cap) w->len++;
}

/*
 * Get current number of tokens in window.
 */

int window_len(const ByteWindow *w) {
    return w->len;
}

/*
 * Get token at a specific age (0 = most recent).
 * 
 * Args:
 *   w - window
 *   age - how many positions back (0 = most recent, 1 = previous, etc.)
 * 
 * Returns: token at that age, or 0 if out of bounds
 */

uint8_t window_get(const ByteWindow *w, int age) {
    if (age < 0 || age >= w->len || w->cap <= 0) return 0;
    
    /* Compute index: pos-1-age, wrapping around if needed */
    int idx = w->pos - 1 - age;
    if (idx < 0) idx += w->cap;
    
    return w->buf[idx];
}
