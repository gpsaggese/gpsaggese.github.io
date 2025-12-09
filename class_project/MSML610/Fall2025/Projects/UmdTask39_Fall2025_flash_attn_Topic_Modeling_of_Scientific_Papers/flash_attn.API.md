# FlashAttention API Documentation

This document provides comprehensive documentation about FlashAttention, its technical implementation, and how to use it through various programming interfaces. FlashAttention is an optimized attention algorithm that reduces memory complexity from O(n²) to O(n) and provides 2-4x speedup compared to standard attention implementations.

## What is FlashAttention?

FlashAttention is an IO-aware exact attention algorithm that reduces memory usage and improves computational speed for transformer models. It was introduced by Dao et al. in 2022 to address the quadratic memory and time complexity limitations of standard attention mechanisms.

### Core Problem

Standard attention mechanisms in transformers compute and store full attention matrices:
- **Memory Complexity**: O(n²) where n is the sequence length
- **Time Complexity**: O(n²) for computing attention scores
- **Bottleneck**: Memory bandwidth limits performance for long sequences or large batch sizes

For example, with sequence length 2048, standard attention requires storing a 2048×2048 attention matrix, consuming significant GPU memory.

### FlashAttention Solution

FlashAttention solves this by:
1. **Tiling**: Splitting the attention computation into smaller blocks that fit in fast GPU memory (SRAM)
2. **Recomputation**: Computing attention on-the-fly without storing full attention matrices
3. **Kernel Fusion**: Fusing attention operations into a single GPU kernel to reduce memory reads/writes
4. **Exact Attention**: Maintaining numerical accuracy (no approximation)

**Key Benefits:**
- **Memory Complexity**: O(n) instead of O(n²)
- **Speed**: 2-4x faster than standard attention on CUDA GPUs
- **Accuracy**: Exact attention computation (no approximation)
- **Scalability**: Enables processing longer sequences and larger batch sizes

## Technical Implementation

### Standard Attention Mechanism

Standard attention computes query (Q), key (K), and value (V) matrices, then calculates attention scores as a matrix multiplication, applies softmax to get attention probabilities, and multiplies with values to produce output. All intermediate matrices (attention scores and probabilities) are stored in GPU memory, requiring O(n²) space.

### FlashAttention Algorithm

FlashAttention uses a tiled approach:
1. **Block-wise Computation**: Split Q, K, V into blocks
2. **Iterative Computation**: Process blocks sequentially
3. **Online Softmax**: Compute softmax incrementally
4. **No Intermediate Storage**: Only store final output

The algorithm maintains numerical stability using online softmax techniques while avoiding storage of full attention matrices.

### Memory and Compute Trade-offs

**Standard Attention:**
- Memory: O(n²) - stores full attention matrix
- Compute: O(n²) - full matrix multiplication
- Memory bandwidth: High (many reads/writes)

**FlashAttention:**
- Memory: O(n) - only stores output
- Compute: O(n²) - same computational complexity
- Memory bandwidth: Low (fewer reads/writes)
- **Net Result**: Faster due to reduced memory I/O

## FlashAttention Requirements

### Hardware Requirements

- **CUDA-Compatible GPU**: FlashAttention requires NVIDIA GPUs with CUDA support
- **Compute Capability**: Requires CUDA compute capability 7.5+ (Turing architecture or newer)
- **GPU Memory**: While FlashAttention reduces memory usage, sufficient GPU memory is still needed for model weights

### Software Requirements

- **PyTorch 2.0+**: For automatic FlashAttention integration via scaled_dot_product_attention
- **CUDA Toolkit**: Compatible CUDA version (typically CUDA 11.6+)
- **FlashAttention Kernel**: Automatically available in PyTorch 2.0+ when conditions are met

### Limitations

- **Not Available on CPU**: FlashAttention requires CUDA GPUs
- **Not Available on MPS (Apple Silicon)**: Only works with NVIDIA GPUs
- **Sequence Length Constraints**: While FlashAttention enables longer sequences, very long sequences may still face limitations
- **Model Compatibility**: Works with most transformer architectures (BERT, RoBERTa, GPT, etc.)

## FlashAttention Integration Methods

### Method 1: PyTorch 2.0+ scaled_dot_product_attention (Recommended)

PyTorch 2.0+ provides automatic FlashAttention integration through the scaled_dot_product_attention function. This method automatically detects CUDA GPU availability and selects the FlashAttention kernel when appropriate, falling back to memory-efficient attention or standard attention otherwise.

**Advantages:**
- Automatic detection and usage
- Graceful fallback to efficient attention on non-CUDA devices
- No additional dependencies
- Works transparently with Hugging Face models

### Method 2: Direct FlashAttention Library (Advanced)

For advanced use cases requiring more control, the direct FlashAttention library can be used. This provides more control over attention computation and supports additional features like ALiBi and block-sparse attention.

**Advantages:**
- More control over attention computation
- Supports additional features (ALiBi, block-sparse attention)

**Limitations:**
- Requires separate installation
- More complex integration
- CUDA-only

### Method 3: Hugging Face Transformers Integration

Modern Hugging Face models automatically use FlashAttention when PyTorch 2.0+ is installed, CUDA GPU is available, and the model is loaded with attn_implementation="sdpa" (which is the default in many models).

## Programming Interfaces for FlashAttention

This section documents how to use FlashAttention through different programming interfaces, from native APIs to custom wrapper functions.

## Native Transformers Library API

The Hugging Face Transformers library provides direct access to transformer models that can use FlashAttention.

### Loading Models with Transformers

The Transformers library uses AutoTokenizer and AutoModel to load pre-trained models. When using PyTorch 2.0+, models automatically use scaled_dot_product_attention, which enables FlashAttention on CUDA GPUs transparently. No code changes are needed - FlashAttention is automatic when available.

### Manual Encoding with Transformers

The Transformers library requires manual tokenization, batching, and embedding extraction. The attention computation in the model forward pass automatically uses FlashAttention on CUDA, providing faster encoding compared to manual attention implementation and enabling larger batch sizes due to reduced memory usage.

### Verifying FlashAttention in Transformers

You can verify FlashAttention is active by checking if PyTorch 2.0+ is installed (scaled_dot_product_attention function exists) and examining the model's attention type, which should show BertSdpaSelfAttention or similar, indicating SDPA/FlashAttention support.

## Native Sentence-Transformers Library API

Sentence-Transformers provides a simplified interface that automatically benefits from FlashAttention.

### Loading Models with Sentence-Transformers

Sentence-Transformers uses underlying Transformer models and automatically benefits from FlashAttention when PyTorch 2.0+ and CUDA are available. No explicit configuration is needed.

### Encoding with Sentence-Transformers

The Sentence-Transformers API provides single-line encoding with automatic FlashAttention optimization, built-in progress tracking, and optimized batching and memory management.

## Custom Wrapper Functions for FlashAttention

These functions from flash_attn_utils.py provide a simplified interface that explicitly manages FlashAttention integration.

### check_flash_attention_support()

Checks if FlashAttention is available via PyTorch 2.0+ scaled_dot_product_attention. Returns a tuple containing a boolean indicating availability and a human-readable status message. This function checks PyTorch version (2.0+ required), CUDA availability (FlashAttention requires CUDA), and scaled_dot_product_attention function availability.

### load_encoder()

Loads a transformer model with FlashAttention support when available. Takes a model name and optional use_flash_attention parameter. Returns tokenizer, model, and device. The function automatically detects CUDA GPU availability, sets attn_implementation="sdpa" when FlashAttention is available, and falls back to standard attention on CPU/MPS. No manual configuration is required.

### encode_texts()

Encodes texts into embeddings, automatically benefiting from FlashAttention if enabled. Provides faster encoding when FlashAttention is active, enables larger batch sizes due to reduced memory usage, and better performance on long sequences. This wrapper provides automatic batching, progress tracking, and FlashAttention optimization, making it more convenient than native Transformers (which requires manual tokenization, batching, pooling) or native Sentence-Transformers (which is simple but offers less control).

### Additional Wrapper Functions

The following wrapper functions complete the topic modeling workflow but don't directly use FlashAttention (they operate on embeddings after encoding):

- **load_and_preprocess_arxiv()**: Loads and preprocesses arXiv data
- **cluster_kmeans()**: Clusters embeddings into topics
- **describe_topics_tfidf()**: Extracts topic keywords
- **reduce_2d()**: Projects embeddings to 2D for visualization

### benchmark_encoding()

Benchmarks encoding performance and reports FlashAttention status. Returns a dictionary containing average encoding time, throughput (documents processed per second), FlashAttention availability status, status message, and device information. This function measures the performance of the current encoding setup and reports whether FlashAttention is active, helping users understand the performance characteristics of their configuration.

## Performance Characteristics

### Expected Speedup

FlashAttention typically provides 2-4x speedup on CUDA GPUs compared to standard attention, with larger speedups for longer sequences and larger batch sizes. Memory savings of 30-50% compared to standard attention are typical.

### Factors Affecting Performance

1. **Sequence Length**: Longer sequences show larger benefits
2. **Batch Size**: Larger batches benefit more from memory efficiency
3. **GPU Model**: Newer GPUs (A100, H100) show better performance
4. **Model Architecture**: Works best with standard transformer attention patterns

### Memory Complexity Comparison

For sequence length n:
- **Standard Attention**: O(n²) memory for attention matrix
- **FlashAttention**: O(n) memory (no attention matrix storage)

Example with n=2048:
- Standard: ~16MB for attention matrix (2048² × 4 bytes)
- FlashAttention: ~8KB for output (2048 × d × 4 bytes, d=embedding_dim)

## When to Use FlashAttention

### Use FlashAttention When:
- Processing large text corpora or long sequences
- GPU memory is limited but you need large batch sizes
- Speed is critical (real-time applications)
- You have CUDA-compatible GPUs available

### Don't Use FlashAttention When:
- Running on CPU or Apple Silicon (MPS)
- Processing very short sequences (overhead may outweigh benefits)
- GPU memory is abundant and speed isn't critical

### Automatic Fallback

The recommended approach (PyTorch 2.0+ SDPA) automatically uses FlashAttention on CUDA GPUs when available, falls back to efficient attention on CUDA when FlashAttention unavailable, and falls back to standard attention on CPU/MPS. No code changes are needed - the system adapts automatically.

## Verification and Testing

### How to Verify FlashAttention is Active

You can verify FlashAttention is active by:
1. **Check Status**: Use check_flash_attention_support() to verify availability
2. **Benchmark Performance**: Use benchmark_encoding() to see FlashAttention status in results
3. **Check Model Configuration**: Verify that the model uses scaled_dot_product_attention (SDPA) which enables FlashAttention on CUDA

### Expected Behavior

**With CUDA GPU:**
- FlashAttention should be available
- Speedup of 2-4x compared to traditional attention
- Embedding accuracy preserved (cosine similarity ~1.0)

**Without CUDA GPU:**
- FlashAttention not available
- Falls back to standard attention
- Still functional, just slower

## Technical Deep Dive

### Attention Computation Flow

**Standard Attention:**
Input goes through Q, K, V computation, then attention matrix calculation (stored in memory), then output production.

**FlashAttention:**
Input goes through Q, K, V blocks, which are processed in tiles with incremental computation, producing output without intermediate storage.

### Online Softmax Algorithm

FlashAttention uses online softmax to compute attention probabilities without storing the full matrix:
1. **Max Value Tracking**: Tracks maximum values across blocks
2. **Exponential Normalization**: Normalizes exponentials incrementally
3. **Numerical Stability**: Maintains precision without full matrix storage

### Memory Access Patterns

**Standard Attention:**
- Reads: Q, K, V matrices from HBM
- Writes: Attention matrix to HBM
- Reads: Attention matrix from HBM
- Writes: Output to HBM
- **Total**: 4 HBM accesses

**FlashAttention:**
- Reads: Q, K, V blocks from HBM to SRAM
- Computes: Attention in SRAM
- Writes: Output blocks to HBM
- **Total**: 2 HBM accesses (50% reduction)
