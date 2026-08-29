# Ternary-MoE Optimization Plan

## Current Issues Analysis

### 1. Generation Quality Problems
- **Root Cause**: Pure argmax sampling (greedy decoding) causes deterministic loops
- **Symptoms**: Repeating patterns like `Vefefefllllll...` 
- **Current Loss**: 22-24 nats (vs theoretical 5.5 for random guessing)
- **Model**: Too small (DIM=32, HID=64, 4 experts), insufficient training

### 2. Training Configuration Issues
- **Demo Epochs**: Only 4+4 (normal+qat) - insufficient for learning structure
- **DATA_MAX**: 16384 bytes - limits corpus size
- **Context Window**: 16 bytes - very limited

### 3. Performance Bottlenecks
- `tern_dot()` is the hottest function (already optimized 4.8x)
- No SIMD vectorization
- No parallelism in training loop
- Memory access patterns could be improved

## Optimization Strategy

### Phase 1: Fix Generation Quality (Priority 1)

#### 1.1 Add Temperature Sampling
```c
// In gen_ctx.h - add sampling functions
int sample_with_temperature(const float *logits, int vocab, float temperature);
int sample_top_k(const float *logits, int vocab, int k, float temperature);
```

#### 1.2 Modify Generation Functions
- Update `train_float_next_two_plane_fast_moe()` and variants
- Add temperature parameter (default: 0.8-1.0)
- Add top-k parameter (default: 50)
- Keep argmax as fallback when temperature=0

#### 1.3 Update CLI Apps
- Add `--temperature` and `--top-k` flags to gen_* apps
- Default to temperature=0.8, top-k=50 for better diversity

### Phase 2: Improve Training Configuration (Priority 2)

#### 2.1 Increase Default Epochs
- Demo: 4+4 → 30+30 epochs
- Allow override via command line

#### 2.2 Increase DATA_MAX
- 16384 → 65536 bytes (64KB)
- Dynamically allocate based on available memory
- Support memory-mapped files for large datasets

#### 2.3 Larger Context Window
- 16 → 64 or 128 bytes
- Configurable at compile time
- Update window.c to support variable context

### Phase 3: Performance Optimizations (Priority 3)

#### 3.1 SIMD Vectorization
- Use ARM NEON / x86 SSE/AVX for tern_dot
- Vectorize softmax and other hot loops
- Runtime CPU feature detection

#### 3.2 Memory Optimization
- Cache-friendly weight layout
- Pre-allocate all memory at startup
- Use aligned allocations for SIMD

#### 3.3 Parallel Training
- Multi-threaded data loading
- Parallel expert computation (MoE)
- Asynchronous I/O for checkpointing

### Phase 4: Advanced Features (Priority 4)

#### 4.1 Adaptive Learning Rate
- Learning rate scheduling
- Gradient clipping

#### 4.2 Better Initialization
- Xavier/Glorot initialization for experts
- Better embedding initialization

#### 4.3 Mixed Precision
- FP16 for forward/backward pass
- FP32 only where necessary

## Implementation Timeline

### Week 1: Core Fixes
- ✅ Temperature sampling
- ✅ Top-k sampling
- ✅ Increase demo epochs to 30+30
- ✅ Increase DATA_MAX to 65536

### Week 2: Performance
- ✅ SIMD tern_dot
- ✅ Memory optimizations
- ✅ Larger context window support

### Week 3: Advanced
- ✅ Parallel MoE computation
- ✅ Learning rate scheduling
- ✅ Documentation and testing

## Expected Results

### Quality Metrics
- Loss: 22-24 nats → < 10 nats (with more training)
- Generation: No repeating patterns, coherent text
- Perplexity: Significant improvement

### Performance Metrics
- Training speed: 2-3x improvement with SIMD
- Memory usage: Optimized, support larger models
- Inference: 40% faster with vectorized tern_dot

## Testing Strategy

1. **Unit Tests**: Add tests for new sampling functions
2. **Integration Tests**: Verify generation quality improvements
3. **Benchmark Tests**: Measure performance improvements
4. **Regression Tests**: Ensure no breaking changes to existing functionality

## Backward Compatibility

- All changes will be backward compatible
- Existing checkpoints remain loadable
- CLI interface extended, not changed
- Default behavior preserved where possible
