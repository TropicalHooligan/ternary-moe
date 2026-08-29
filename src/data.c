#include "data.h"
#include <stdio.h>

/*
 * ============================================================================
 * Data Loading and Sampling
 * ============================================================================
 */

/*
 * Load data from a file.
 * 
 * Args:
 *   buf - buffer to load into
 *   max_len - maximum bytes to read
 *   path - file path
 * 
 * Returns: number of bytes read, or -1 on error
 */

int data_load_file(uint8_t *buf, int max_len, const char *path) {
    if (max_len <= 0) return 0;
    
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    
    size_t r = fread(buf, 1, (size_t)max_len, f);
    fclose(f);
    
    return (int)r;
}

/*
 * Create a sample from the data at a specific position.
 * 
 * Args:
 *   data - data buffer
 *   len - data length
 *   pos - position to sample from
 *   w - window for context
 *   ctx - context length
 * 
 * Returns: target token at position pos, or -1 on error
 */

int data_make_sample(const uint8_t *data, int len, int pos, ByteWindow *w, int ctx) {
    if (!data || !w || pos < 0 || pos >= len) return -1;
    
    /* Reset window */
    window_reset(w, w->buf, w->cap);
    
    /* Fill window with context tokens */
    int start = pos - ctx;
    if (start < 0) start = 0;
    
    for (int i = start; i < pos; ++i) {
        window_push(w, data[i]);
    }
    
    /* Return target token */
    return data[pos];
}
