# FlashAttention API

This document describes the native programming interfaces for FlashAttention-enabled topic modeling using Transformers and Sentence-Transformers libraries, along with our custom wrapper layer. For hands-on examples and code demonstrations, see [`flash_attn.API.ipynb`](flash_attn.API.ipynb).

## Description

This document explains the design and architecture of:

- **Native Transformers library API**: Direct usage of Hugging Face Transformers for loading models and encoding text
- **Native Sentence-Transformers library API**: Simplified text encoding using optimized sentence embedding models
- **FlashAttention integration**: PyTorch 2.0+ integration via `scaled_dot_product_attention` for efficient attention computation
- **Custom wrapper layer**: Utility functions from `flash_attn_utils.py` that simplify common topic modeling workflows

## Overview

Our integration layer provides a streamlined approach to encoding text data into embeddings using FlashAttention-optimized transformers, clustering embeddings into topics, and visualizing results. This helps researchers and developers to:

1. Efficiently encode large text corpora using FlashAttention-optimized models
2. Automatically handle device selection (CUDA, MPS, CPU) and memory management
3. Cluster embeddings into distinct topics using K-Means
4. Extract and label topics using TF-IDF analysis
5. Visualize topic distributions using dimensionality reduction techniques

## Problem Statement

While modern transformer models provide powerful text encoding capabilities, using them directly for topic modeling presents several challenges:

- **Complexity**: Managing model loading, tokenization, batching, and attention computation requires significant boilerplate code.
- **Memory Management**: Standard attention mechanisms have quadratic memory complexity, making them inefficient for long sequences or large batch sizes.
- **Device Handling**: Manually managing CUDA, MPS, and CPU device placement across different systems is error-prone.
- **Performance**: Without FlashAttention optimization, encoding large text corpora can be slow and memory-intensive.
- **Integration**: Combining embeddings, clustering, and visualization requires coordinating multiple libraries (transformers, scikit-learn, UMAP).

To overcome these limitations, our integration layer provides:

- Simplified API functions that abstract away device management and batch processing.
- FlashAttention integration for efficient memory usage and faster computation.
- Automated embedding extraction with proper pooling and normalization.
- Integrated clustering and topic labeling workflows.
- Ready-to-use visualization utilities.

## Alternatives and Comparisons

### Native Transformers Library (Hugging Face)

**Advantages:**
- Comprehensive model support with thousands of pre-trained models.
- Flexible API for fine-tuning and custom architectures.
- Active community and extensive documentation.

**Limitations:**
- Standard attention mechanisms have O(n²) memory complexity.
- Manual device management required (CUDA/MPS/CPU).
- Embedding extraction requires custom pooling logic.
- No built-in topic modeling capabilities.

### Native FlashAttention Library

**Advantages:**
- Linear memory complexity instead of quadratic.
- Significant speedup (2-4x) over standard attention implementations.
- Exact attention computation (no approximation).

**Limitations:**
- Requires CUDA-compatible GPU (not available on CPU/MPS).
- Complex installation process.
- Low-level API requiring custom kernel implementations.

### PyTorch 2.0+ `scaled_dot_product_attention` with FlashAttention Backend

**Advantages:**
- Automatic FlashAttention usage when available (CUDA).
- Falls back to standard attention on CPU/MPS.
- Easy to integrate into existing models.
- No additional dependencies beyond PyTorch 2.0+.

**Limitations:**
- Requires PyTorch 2.0+.
- FlashAttention backend requires compatible CUDA environment.

### Recommendation

For our topic modeling use case, **PyTorch 2.0+ with `scaled_dot_product_attention`** is the best option because it:

- Provides FlashAttention benefits when available (CUDA GPUs).
- Gracefully falls back to standard attention on other devices.
- Requires minimal code changes.
- Works across different hardware configurations.

## Native API Overview

### Transformers Library (Hugging Face)

The Transformers library provides access to pre-trained transformer models:

- **AutoTokenizer**: Automatically loads appropriate tokenizers for models.
- **AutoModel**: Loads pre-trained transformer models (BERT, RoBERTa, etc.).
- **Model Architecture**: Access to `last_hidden_state` for embeddings.
- **Attention Mechanisms**: Built-in attention computation (standard or FlashAttention if supported).

### Sentence-Transformers Library

Sentence-Transformers provides models optimized for semantic similarity:

- **Pre-trained Models**: Models like `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and `intfloat/e5-large-v2` (used in the main example notebook)
- **Semantic Embeddings**: Dense vector representations capturing semantic meaning
- **Optimized for Sentence-level Tasks**: Better suited for topic modeling than base transformers

### FlashAttention Integration

FlashAttention can be integrated via:

1. **PyTorch 2.0+ `torch.nn.functional.scaled_dot_product_attention`**:
   - Automatically uses FlashAttention kernel when available.
   - Falls back to efficient attention implementation otherwise.
   - Works transparently with Hugging Face models.

2. **Direct FlashAttention Library** (for advanced use cases):
   - Requires CUDA-compatible GPU.
   - Provides more control over attention computation.
   - More complex setup and integration.

### Challenges with Native APIs

- **Manual Attention Configuration**: Enabling FlashAttention requires model architecture modifications.
- **Device Management**: Explicit device placement needed for CUDA/MPS/CPU.
- **Batch Processing**: Manual batching logic required for large datasets.
- **Embedding Extraction**: Custom pooling strategies (mean, CLS token, etc.) needed.
- **Memory Efficiency**: Large batches can cause out-of-memory errors without optimization.

## Our Integration Layer

### Goals

1. **Simplified Model Loading**: Automatic device detection and model configuration.
2. **FlashAttention Support**: Seamless integration with FlashAttention when available.
3. **Efficient Encoding**: Optimized batch processing with progress tracking.
4. **Complete Workflow**: End-to-end pipeline from text to topics to visualization.
5. **Cross-Platform Compatibility**: Works on CUDA, MPS, and CPU without code changes.

### Core Functions

#### 1. `load_encoder(model_name: str = "intfloat/e5-base-v2", use_flash_attention: bool = True) -> Tuple[AutoTokenizer, AutoModel, torch.device]`

Loads a transformer model and tokenizer with automatic device selection and optional FlashAttention support.

- Handles model downloading from Hugging Face Hub.
- Automatically detects and selects appropriate device (CUDA > MPS > CPU).
- Configures FlashAttention when `use_flash_attention=True` and CUDA is available.
- Falls back to standard attention on CPU/MPS or when FlashAttention is disabled.
- Configures model for evaluation mode.
- Returns tokenizer, model, and device for consistent usage.

**Parameters:**
- `model_name`: Hugging Face model identifier (default: "intfloat/e5-base-v2").
- `use_flash_attention`: Whether to enable FlashAttention via SDPA when available (default: True).

#### 2. `encode_texts(texts: Iterable[str], tokenizer, model, device, batch_size=64, max_length=256) -> np.ndarray`

Encodes a collection of texts into dense embeddings.

- Automatic batching with progress tracking.
- Efficient tokenization with padding and truncation.
- Mean pooling over sequence length (masked).
- Returns NumPy array of embeddings (n_docs × embedding_dim).

#### 3. `load_and_preprocess_arxiv(path: Union[str, Path], max_docs: Optional[int] = None, min_abstract_len: int = 50) -> pd.DataFrame`

Loads and preprocesses arXiv metadata JSONL file.

- Streaming JSONL parser for memory efficiency.
- Text preprocessing (title + abstract concatenation).
- Filtering of short or invalid abstracts.
- Returns cleaned DataFrame with id, title, abstract, categories, text.

#### 4. `cluster_kmeans(embeddings: np.ndarray, k: int = 20, random_state: int = 42) -> Tuple[np.ndarray, KMeans]`

Clusters embeddings using K-Means algorithm.

- Deterministic clustering with random seed.
- Returns cluster labels and fitted KMeans model.

#### 5. `describe_topics_tfidf(texts: List[str], labels: np.ndarray, top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]`

Extracts topic keywords using TF-IDF analysis.

- Class-based TF-IDF for topic-specific term extraction.
- Supports n-grams (unigrams and bigrams).
- Returns top keywords per topic with TF-IDF scores.

#### 6. `reduce_2d(embeddings: np.ndarray, method: str, random_state: int) -> np.ndarray`

Projects high-dimensional embeddings to 2D for visualization.

- UMAP or t-SNE dimensionality reduction.
- Configurable hyperparameters.
- Returns 2D coordinates (n_docs × 2).

#### 7. `check_flash_attention_support() -> Tuple[bool, str]`

Checks if FlashAttention is available via PyTorch 2.0+ scaled_dot_product_attention.

- **Returns**: Tuple of (is_available, description)
- **Detects**: PyTorch version, CUDA availability, FlashAttention support
- **Usage**: Check FlashAttention availability before model loading

#### 8. `benchmark_encoding(texts, tokenizer, model, device, batch_size=64, max_length=256, warmup=1, trials=3) -> Dict[str, Any]`

Benchmarks encoding performance with timing metrics and FlashAttention status.

- **Args**: 
  - `texts`: List of texts to encode
  - `tokenizer`: Tokenizer instance
  - `model`: Model instance
  - `device`: Device to run on
  - `batch_size`: Batch size for encoding
  - `max_length`: Maximum sequence length
  - `warmup`: Number of warmup runs
  - `trials`: Number of benchmark trials
- **Returns**: Dictionary with performance metrics (throughput, latency, FlashAttention status)
- **Metrics**: Average time, throughput (docs/sec), device, FlashAttention availability
- **Usage**: Compare performance with/without FlashAttention

### FlashAttention Integration Strategy

Our wrapper uses PyTorch 2.0+ `scaled_dot_product_attention` which:

1. **Automatically enables FlashAttention** when:
   - PyTorch 2.0+ is installed.
   - CUDA-compatible GPU is available.
   - FlashAttention kernel is available.

2. **Falls back gracefully** to:
   - Memory-efficient attention on CUDA (if FlashAttention unavailable).
   - Standard attention on CPU/MPS.

3. **No code changes required**:
   - Works transparently with Hugging Face models.
   - Model forward pass automatically uses optimized attention.

This approach ensures maximum performance on supported hardware while maintaining compatibility across all platforms.

## Conclusion

This integration layer enhances the usability of FlashAttention and transformer models for topic modeling by:

- Abstracting away complexity of device management and model configuration.
- Providing efficient encoding with automatic batching and progress tracking.
- Enabling FlashAttention benefits when available (2-4x speedup, linear memory).
- Delivering a complete workflow from raw text to visualized topics.
- Maintaining cross-platform compatibility for diverse hardware configurations.

The layer simplifies development, improves performance, and enables researchers to focus on topic modeling insights rather than low-level implementation details.

## Citations

- **FlashAttention**: Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." https://arxiv.org/abs/2205.14135
- **E5 Embeddings**: Wang et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." https://arxiv.org/abs/2212.03533
- **Transformers**: Wolf et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." https://arxiv.org/abs/1910.03771

