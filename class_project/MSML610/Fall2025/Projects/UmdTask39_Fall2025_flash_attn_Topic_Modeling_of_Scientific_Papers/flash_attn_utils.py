# flash_attn_utils.py (Python 3.8-compatible typing)

import json, re, logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_whitespace_re = re.compile(r"\s+")

def stream_arxiv_jsonl(path: Union[str, Path], max_docs: Optional[int] = None):
    """
    Stream a large JSONL (one record per line). Yields a minimal dict per paper.
    """
    path = Path(path)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {
                "id": rec.get("id"),
                "title": rec.get("title"),
                "abstract": rec.get("abstract"),
                "categories": rec.get("categories"),
            }
            n += 1
            if max_docs is not None and n >= max_docs:
                break

def preprocess_record(rec: Dict[str, Any], min_abstract_len: int = 50) -> Optional[Dict[str, Any]]:
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    if not title or not abstract or len(abstract) < min_abstract_len:
        return None
    text = f"{title}. {abstract}"
    text = _whitespace_re.sub(" ", text).lower()
    return {
        "id": rec.get("id"),
        "title": title,
        "abstract": abstract,
        "categories": rec.get("categories"),
        "text": text,
    }

def load_and_preprocess_arxiv(path: Union[str, Path], max_docs: Optional[int] = None, min_abstract_len: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for raw in stream_arxiv_jsonl(path, max_docs=max_docs):
        cleaned = preprocess_record(raw, min_abstract_len=min_abstract_len)
        if cleaned is not None:
            rows.append(cleaned)
    return pd.DataFrame.from_records(rows)

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def check_flash_attention_support() -> Tuple[bool, str]:
    """
    Check if FlashAttention is available via PyTorch 2.0+ scaled_dot_product_attention.
    
    Returns:
        Tuple[bool, str]: (is_available, description)
    """
    has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    
    if not has_sdpa:
        return False, "PyTorch < 2.0: scaled_dot_product_attention not available"
    
    # Check if on CUDA (FlashAttention requires CUDA)
    if torch.cuda.is_available():
        return True, "FlashAttention available via scaled_dot_product_attention (CUDA)"
    else:
        return False, "scaled_dot_product_attention available but FlashAttention requires CUDA (running on CPU/MPS)"

def load_encoder(
    model_name: str = "intfloat/e5-base-v2",
    use_flash_attention: bool = True
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """
    Load a transformer model with optional FlashAttention optimization.
    
    Args:
        model_name: Hugging Face model identifier
        use_flash_attention: Whether to enable FlashAttention via SDPA when available
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    
    # Determine attention implementation
    attn_implementation = None
    if use_flash_attention:
        has_flash, flash_msg = check_flash_attention_support()
        if has_flash:
            # Use SDPA which will automatically use FlashAttention on CUDA
            attn_implementation = "sdpa"
            logger.info("FlashAttention: %s", flash_msg)
        else:
            logger.info("FlashAttention: %s (falling back to default attention)", flash_msg)
            attn_implementation = "eager"
    else:
        # Explicitly disable FlashAttention/SDPA - use eager (traditional) attention
        attn_implementation = "eager"
        logger.info("FlashAttention explicitly disabled - using traditional attention (eager)")
    
    # Load model with attention implementation setting
    # Try to set attn_implementation regardless of config check
    # Many models accept this parameter even if not in config
    try:
        # First, try loading with attn_implementation parameter directly
        if attn_implementation:
            try:
                mdl = AutoModel.from_pretrained(
                    model_name,
                    attn_implementation=attn_implementation
                )
                logger.info("Successfully set attn_implementation=%s", attn_implementation)
            except TypeError:
                # Parameter not supported, try via config
                config = AutoConfig.from_pretrained(model_name)
                try:
                    # Try to modify config before loading
                    if hasattr(config, 'attn_implementation'):
                        config.attn_implementation = attn_implementation
                    mdl = AutoModel.from_pretrained(model_name, config=config)
                    logger.info("Set attn_implementation via config: %s", attn_implementation)
                except Exception as e2:
                    # Fall back to default loading
                    logger.warning("Could not set attn_implementation: %s. Using default.", e2)
                    mdl = AutoModel.from_pretrained(model_name)
        else:
            mdl = AutoModel.from_pretrained(model_name)
    except Exception as e:
        logger.warning("Could not set attention implementation: %s. Using default.", e)
        mdl = AutoModel.from_pretrained(model_name)
    
    mdl.to(device).eval()
    
    # Verify attention mechanism after loading
    actual_attn = None
    if hasattr(mdl.config, 'attn_implementation'):
        actual_attn = mdl.config.attn_implementation
    elif hasattr(mdl.config, '_attn_implementation'):
        actual_attn = mdl.config._attn_implementation
    
    if actual_attn:
        logger.info("Model loaded with attn_implementation: %s", actual_attn)
    else:
        logger.info("Model loaded - attn_implementation not explicitly set in config (may use defaults)")
    
    if device.type == "cuda" and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.info("Model loaded with potential FlashAttention support (PyTorch 2.0+ SDPA on CUDA)")
    
    logger.info("Loaded model %s on %s", model_name, device)
    return tok, mdl, device

@torch.no_grad()
def encode_texts(
    texts: Iterable[str],
    tok: AutoTokenizer,
    mdl: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
) -> np.ndarray:
    texts = list(texts)
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc)
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs)

def cluster_kmeans(embeddings: np.ndarray, k: int = 20, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(embeddings)
    return labels, km

def describe_topics_tfidf(texts: List[str], labels: np.ndarray, top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1,2), min_df=5)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    topics: Dict[int, List[Tuple[str, float]]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = X[idx].mean(axis=0).A1
        top = centroid.argsort()[::-1][:top_k]
        topics[int(c)] = list(zip(vocab[top], centroid[top].tolist()))
    return topics

def reduce_2d(embeddings: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    if method == "tsne":
        proj = TSNE(n_components=2, init="pca", random_state=random_state, perplexity=30, learning_rate="auto")
        return proj.fit_transform(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)

def benchmark_encoding(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    warmup: int = 1,
    trials: int = 3
) -> Dict[str, Any]:
    """
    Benchmark encoding performance with timing metrics.
    
    Args:
        texts: List of texts to encode
        tokenizer: Tokenizer instance
        model: Model instance
        device: Device to run on
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        warmup: Number of warmup runs
        trials: Number of benchmark trials
        
    Returns:
        Dictionary with performance metrics (throughput, latency, etc.)
    """
    import time
    
    # Warmup runs
    for _ in range(warmup):
        _ = encode_texts(texts[:min(100, len(texts))], tokenizer, model, device, batch_size, max_length)
    
    # Benchmark trials
    times = []
    for _ in range(trials):
        start = time.time()
        _ = encode_texts(texts, tokenizer, model, device, batch_size, max_length)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    throughput = len(texts) / avg_time if avg_time > 0 else 0
    
    # Check if FlashAttention is being used
    has_flash, flash_msg = check_flash_attention_support()
    
    return {
        "avg_time_seconds": avg_time,
        "throughput_docs_per_sec": throughput,
        "num_texts": len(texts),
        "batch_size": batch_size,
        "flash_attention_available": has_flash,
        "flash_attention_status": flash_msg,
        "device": str(device)
    }

def explain_topic_assignment(
    paper_text: str,
    paper_embedding: np.ndarray,
    topic_id: int,
    topic_centroid: np.ndarray,
    topic_keywords: List[str],
    all_texts: List[str],
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    top_k_similar: int = 5,
    top_k_keywords: int = 10
) -> Dict[str, Any]:
    """
    Explain why a paper belongs to a specific topic.
    
    Args:
        paper_text: Text content of the paper (title + abstract)
        paper_embedding: Embedding vector of the paper
        topic_id: Topic ID to explain assignment to
        topic_centroid: Centroid of the topic cluster
        topic_keywords: Keywords for the topic
        all_texts: All texts in the dataset
        all_embeddings: All embeddings in the dataset
        all_labels: All cluster labels
        top_k_similar: Number of similar papers to show
        top_k_keywords: Number of matching keywords to show
    
    Returns:
        Dictionary with explanation details
    """
    import re
    
    # 1. Calculate distance to topic centroid
    distance = np.linalg.norm(paper_embedding - topic_centroid)
    
    # 2. Find matching keywords in paper text
    paper_lower = paper_text.lower()
    matching_keywords = []
    keyword_scores = {}
    
    for keyword in topic_keywords[:top_k_keywords]:
        keyword_clean = keyword.lower().replace("_", " ")
        # Check if keyword appears in paper (as word, not substring)
        pattern = r'\b' + re.escape(keyword_clean) + r'\b'
        if re.search(pattern, paper_lower):
            matching_keywords.append(keyword)
            # Simple frequency score
            count = len(re.findall(pattern, paper_lower))
            keyword_scores[keyword] = count
    
    # 3. Find similar papers in same topic
    topic_mask = all_labels == topic_id
    topic_indices = np.where(topic_mask)[0]
    topic_embeddings = all_embeddings[topic_indices]
    
    # Calculate cosine similarity to all papers in topic
    dot_product = np.dot(topic_embeddings, paper_embedding)
    norms = np.linalg.norm(topic_embeddings, axis=1) * np.linalg.norm(paper_embedding)
    similarities = dot_product / (norms + 1e-8)
    
    # Get top K similar papers
    top_indices = similarities.argsort()[::-1][:top_k_similar]
    similar_papers = []
    for idx in top_indices:
        original_idx = topic_indices[idx]
        similar_papers.append({
            "index": int(original_idx),
            "similarity": float(similarities[idx]),
            "title": all_texts[original_idx][:100] + "..." if len(all_texts[original_idx]) > 100 else all_texts[original_idx]
        })
    
    # 4. Generate human-readable explanation
    matching_count = len(matching_keywords)
    top_matches = matching_keywords[:5]
    
    explanation = f"""
    Paper Assignment to Topic {topic_id}:
    
    Distance to Topic Centroid: {distance:.4f}
    (Lower distance = better fit to topic)
    
    Matching Keywords: {matching_count} out of {len(topic_keywords)} topic keywords
    Top matches: {', '.join(top_matches) if top_matches else 'None'}
    
    Similar Papers in Same Topic: {len(similar_papers)} papers
    (Top {top_k_similar} most similar papers shown)
    
    Rationale: This paper is assigned to Topic {topic_id} because:
    - It is {distance:.2f} units away from the topic centroid
    - It contains {matching_count} topic-specific keywords
    - It is semantically similar to other papers in this topic
    """
    
    return {
        "topic_id": topic_id,
        "distance_to_centroid": float(distance),
        "matching_keywords": matching_keywords,
        "keyword_scores": keyword_scores,
        "similar_papers": similar_papers,
        "explanation": explanation,
        "matching_keyword_count": matching_count
    }

def compare_flash_attention_vs_traditional(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_length: int = 256,
    warmup: int = 1,
    trials: int = 3,
    max_sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare FlashAttention vs traditional attention performance and accuracy.
    
    Args:
        texts: List of texts to encode
        model_name: Model identifier
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        warmup: Number of warmup runs
        trials: Number of benchmark trials
        max_sample_size: Maximum number of texts to use for comparison (None = use all)
    
    Returns:
        Dictionary with comparison results
    """
    import time
    import gc
    
    # Use all texts by default, or limit if max_sample_size is specified
    if max_sample_size is not None:
        sample_size = min(max_sample_size, len(texts))
        sample_texts = texts[:sample_size]
    else:
        sample_texts = texts
    
    # ===== Test 1: FlashAttention Enabled =====
    print("Testing with FlashAttention enabled...")
    tok_flash, mdl_flash, device_flash = load_encoder(model_name, use_flash_attention=True)
    
    # Check FlashAttention model config - improved method
    flash_attn_config = None
    if hasattr(mdl_flash.config, 'attn_implementation'):
        flash_attn_config = mdl_flash.config.attn_implementation
    elif hasattr(mdl_flash.config, '_attn_implementation'):
        flash_attn_config = mdl_flash.config._attn_implementation
    
    if flash_attn_config:
        print(f"FlashAttention model config: {flash_attn_config}")
    else:
        print(f"FlashAttention model config: Not explicitly set (may default to SDPA)")
        # Try to check via model attributes
        if hasattr(mdl_flash, 'encoder') and hasattr(mdl_flash.encoder, 'layer'):
            first_layer = mdl_flash.encoder.layer[0]
            if hasattr(first_layer, 'attention'):
                print(f"  Model architecture: {type(first_layer.attention).__name__}")
    
    # Warmup
    for _ in range(warmup):
        _ = encode_texts(sample_texts[:min(100, len(sample_texts))], tok_flash, mdl_flash, device_flash, batch_size, max_length)
    
    # Benchmark
    times_flash = []
    for _ in range(trials):
        gc.collect()
        if device_flash.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start = time.time()
        emb_flash = encode_texts(sample_texts, tok_flash, mdl_flash, device_flash, batch_size, max_length)
        if device_flash.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times_flash.append(elapsed)
    
    avg_time_flash = sum(times_flash) / len(times_flash)
    throughput_flash = len(sample_texts) / avg_time_flash
    
    # Memory usage (GPU)
    if device_flash.type == "cuda" and torch.cuda.is_available():
        memory_allocated_flash = torch.cuda.memory_allocated(device_flash.index) / (1024**2)
        memory_reserved_flash = torch.cuda.memory_reserved(device_flash.index) / (1024**2)
    else:
        memory_allocated_flash = 0
        memory_reserved_flash = 0
    
    # Clean up
    del mdl_flash, tok_flash
    gc.collect()
    if device_flash.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ===== Test 2: Traditional Attention =====
    print("Testing with traditional attention (FlashAttention disabled)...")
    tok_trad, mdl_trad, device_trad = load_encoder(model_name, use_flash_attention=False)
    
    # Check Traditional model config - improved method
    trad_attn_config = None
    if hasattr(mdl_trad.config, 'attn_implementation'):
        trad_attn_config = mdl_trad.config.attn_implementation
    elif hasattr(mdl_trad.config, '_attn_implementation'):
        trad_attn_config = mdl_trad.config._attn_implementation
    
    if trad_attn_config:
        print(f"Traditional model config: {trad_attn_config}")
    else:
        print(f"Traditional model config: Not explicitly set (may default to SDPA)")
        # Try to check via model attributes
        if hasattr(mdl_trad, 'encoder') and hasattr(mdl_trad.encoder, 'layer'):
            first_layer = mdl_trad.encoder.layer[0]
            if hasattr(first_layer, 'attention'):
                print(f"  Model architecture: {type(first_layer.attention).__name__}")
    print("-" * 70)
    
    # Warmup
    for _ in range(warmup):
        _ = encode_texts(sample_texts[:min(100, len(sample_texts))], tok_trad, mdl_trad, device_trad, batch_size, max_length)
    
    # Benchmark
    times_trad = []
    for _ in range(trials):
        gc.collect()
        if device_trad.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start = time.time()
        emb_trad = encode_texts(sample_texts, tok_trad, mdl_trad, device_trad, batch_size, max_length)
        if device_trad.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times_trad.append(elapsed)
    
    avg_time_trad = sum(times_trad) / len(times_trad)
    throughput_trad = len(sample_texts) / avg_time_trad
    
    # Memory usage (GPU)
    if device_trad.type == "cuda" and torch.cuda.is_available():
        memory_allocated_trad = torch.cuda.memory_allocated(device_trad.index) / (1024**2)
        memory_reserved_trad = torch.cuda.memory_reserved(device_trad.index) / (1024**2)
    else:
        memory_allocated_trad = 0
        memory_reserved_trad = 0
    
    # ===== Calculate Embedding Similarity =====
    if len(emb_flash.shape) > 2:
        emb_flash = emb_flash.reshape(-1, emb_flash.shape[-1])
    if len(emb_trad.shape) > 2:
        emb_trad = emb_trad.reshape(-1, emb_trad.shape[-1])
    
    emb_flash_norm = emb_flash / (np.linalg.norm(emb_flash, axis=1, keepdims=True) + 1e-8)
    emb_trad_norm = emb_trad / (np.linalg.norm(emb_trad, axis=1, keepdims=True) + 1e-8)
    
    cosine_similarities = np.sum(emb_flash_norm * emb_trad_norm, axis=1)
    avg_similarity = float(np.mean(cosine_similarities))
    min_similarity = float(np.min(cosine_similarities))
    max_similarity = float(np.max(cosine_similarities))
    
    # ===== Calculate Speedup =====
    speedup = avg_time_trad / avg_time_flash if avg_time_flash > 0 else 1.0
    speedup_percent = ((avg_time_trad - avg_time_flash) / avg_time_trad * 100) if avg_time_trad > 0 else 0.0
    
    # Memory reduction
    if memory_allocated_trad > 0:
        memory_reduction = ((memory_allocated_trad - memory_allocated_flash) / memory_allocated_trad * 100)
    else:
        memory_reduction = 0.0
    
    return {
        "flash_attention": {
            "avg_time_seconds": avg_time_flash,
            "throughput_docs_per_sec": throughput_flash,
            "memory_allocated_mb": memory_allocated_flash,
            "memory_reserved_mb": memory_reserved_flash,
            "enabled": True,
            "attn_config": flash_attn_config if flash_attn_config else "default"
        },
        "traditional_attention": {
            "avg_time_seconds": avg_time_trad,
            "throughput_docs_per_sec": throughput_trad,
            "memory_allocated_mb": memory_allocated_trad,
            "memory_reserved_mb": memory_reserved_trad,
            "enabled": False,
            "attn_config": trad_attn_config if trad_attn_config else "default"
        },
        "speedup": {
            "ratio": speedup,
            "percent_faster": speedup_percent,
            "time_saved_seconds": avg_time_trad - avg_time_flash
        },
        "accuracy": {
            "avg_cosine_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "note": "Similarity of 1.0 means identical embeddings (FlashAttention preserves accuracy)"
        },
        "memory": {
            "reduction_percent": memory_reduction,
            "saved_mb": memory_allocated_trad - memory_allocated_flash
        },
        "num_texts": len(sample_texts),
        "device": str(device_flash)
    }