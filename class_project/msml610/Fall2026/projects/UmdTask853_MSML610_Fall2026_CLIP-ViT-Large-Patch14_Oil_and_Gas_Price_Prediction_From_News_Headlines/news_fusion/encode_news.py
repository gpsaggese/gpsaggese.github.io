#!/usr/bin/env python3
"""Frozen CLIP, FinBERT and optional MiniLM headline embeddings with resumable cache.

Outputs one NPZ per encoder with aligned headline IDs. Missing cache shards are
encoded; cached shards are verified. This is REAL model inference, not synthetic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

EVENT_PROMPTS = {
    "wti": ["oil supply disruption", "OPEC announces an oil production cut",
            "oil demand increases", "oil demand weakens", "oil refinery outage",
            "oil exports restricted by sanctions"],
    "henry_hub": ["natural gas storage drawdown", "natural gas inventory build",
                  "cold weather increases natural gas demand", "warm weather reduces gas heating demand",
                  "natural gas pipeline disruption", "LNG exports increase natural gas demand"],
}

CHECKPOINT = {"clip": "openai/clip-vit-large-patch14",
              "finbert": "ProsusAI/finbert",
              "minilm": "sentence-transformers/all-MiniLM-L6-v2"}


def cache_prefix(ids, texts, encoder, checkpoint, revision):
    payload = json.dumps([ids, texts, encoder, checkpoint, revision], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def model_and_tokenizer(encoder, checkpoint, revision, device):
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    if encoder == "clip":
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(checkpoint, revision=revision)
    elif encoder == "finbert":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision)
        labels = [str(v).lower() for v in model.config.id2label.values()]
        if not all(x in labels for x in ("positive", "negative", "neutral")):
            raise ValueError("FinBERT labels do not contain positive/negative/neutral")
    elif encoder == "minilm":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(checkpoint, revision=revision)
    else:
        raise ValueError(encoder)
    model.requires_grad_(False)
    model.eval().to(device)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Encoder MUST be frozen")
    return model, tokenizer


def encode_batch(texts, *, encoder, model, tokenizer, device):
    import torch
    max_length = 77 if encoder == "clip" else (128 if encoder == "finbert" else 256)
    x = tokenizer(texts, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
    x = {k: v.to(device) for k, v in x.items()}
    with torch.inference_mode():
        if encoder == "clip":
            output = model.get_text_features(**x)
            if not isinstance(output, torch.Tensor):
                raise TypeError("CLIP text_features API returned unexpected type; check transformers version")
            vectors = torch.nn.functional.normalize(output.float(), p=2, dim=-1)
            sentiment = None
        elif encoder == "finbert":
            output = model(**x, output_hidden_states=True, return_dict=True)
            vectors = output.hidden_states[-1][:, 0, :].float()
            probabilities = torch.softmax(output.logits.float(), dim=-1)
            # Checkpoint label ordering is read from its config, never assumed.
            names = {str(label).lower(): int(index)
                     for index, label in model.config.id2label.items()}
            sentiment = probabilities[:, [names["positive"], names["negative"], names["neutral"]]]
        else:
            output = model(**x, return_dict=True)
            hidden = output.last_hidden_state.float()
            mask = x["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
            sentiment = None
    return vectors.cpu().numpy().astype("float32"), (
        sentiment.cpu().numpy().astype("float32") if sentiment is not None else None)


def save_npz(path: Path, *, ids, vectors, sentiment=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as h:
        if sentiment is None:
            np.savez_compressed(h, ids=np.asarray(ids, dtype=str), vectors=vectors)
        else:
            np.savez_compressed(h, ids=np.asarray(ids, dtype=str), vectors=vectors,
                                sentiment=sentiment)
    tmp.replace(path)


def encode_manifest(manifest: pd.DataFrame, *, encoder: str, output: Path,
                    cache_dir: Path, batch_size: int = 16,
                    shard_size: int = 256, device: str = "auto", revision: str = "main") -> dict:
    if encoder not in CHECKPOINT:
        raise ValueError(f"Encoder must be in {list(CHECKPOINT)}")
    if batch_size < 1 or shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")
    ids = manifest.headline_id.astype(str).tolist()
    texts = manifest.headline_text.astype(str).tolist()
    if not len(texts) or any(not t.strip() for t in texts):
        raise ValueError("No headlines / empty input")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate headline IDs")
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers before encoding") from exc
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    torch.manual_seed(42)
    checkpoint = CHECKPOINT[encoder]
    key = cache_prefix(ids, texts, encoder, checkpoint, revision)
    cached = cache_dir / f"{encoder}_{key}"
    cached.mkdir(parents=True, exist_ok=True)
    slices = [(i, min(len(texts), i + shard_size)) for i in range(0, len(texts), shard_size)]
    model = tokenizer = None
    arrays = []
    scores = []
    actual_revision = None
    try:
        for k, (i, j) in enumerate(slices, 1):
            dest = cached / f"{i:08d}_{j:08d}.npz"
            if not dest.exists():
                if model is None:
                    print(f"[{encoder}] loading frozen {checkpoint}@{revision}; device={selected_device}", flush=True)
                    model, tokenizer = model_and_tokenizer(encoder, checkpoint, revision, selected_device)
                    actual_revision = getattr(model.config, "_commit_hash", None)
                vecs, probs = [], []
                for m in range(i, j, batch_size):
                    v, p = encode_batch(texts[m:min(j, m + batch_size)], encoder=encoder,
                                        model=model, tokenizer=tokenizer, device=selected_device)
                    vecs.append(v)
                    if p is not None:
                        probs.append(p)
                vectors = np.vstack(vecs)
                sentiments = np.vstack(probs) if probs else None
                save_npz(dest, ids=ids[i:j], vectors=vectors, sentiment=sentiments)
            with np.load(dest, allow_pickle=False) as shard:
                if shard["ids"].astype(str).tolist() != ids[i:j]:
                    raise ValueError(f"Mismatched cached IDs: {dest}")
                arr = shard["vectors"].astype("float32")
                if arr.shape[0] != j-i or not np.isfinite(arr).all():
                    raise ValueError(f"Corrupt vectors: {dest}")
                arrays.append(arr)
                if encoder == "finbert":
                    if "sentiment" not in shard:
                        raise ValueError(f"Missing sentiment: {dest}")
                    p = shard["sentiment"].astype("float32")
                    if p.shape != (j-i, 3) or not np.allclose(p.sum(axis=1), 1., atol=1e-4):
                        raise ValueError(f"Corrupt FinBERT probabilities: {dest}")
                    scores.append(p)
            print(f"[{encoder}] shard {k}/{len(slices)}, records through {j}/{len(texts)}", flush=True)
        # Encode a fixed, predeclared commodity-event prompt vocabulary with CLIP.
        # This creates a low-dimensional event-score control alongside PCA embeddings.
        if encoder == "clip":
            pids = [f"{asset}_{i:02d}" for asset, phrases in EVENT_PROMPTS.items()
                    for i, _ in enumerate(phrases)]
            ptexts = [phrase for phrases in EVENT_PROMPTS.values() for phrase in phrases]
            pkey = cache_prefix(pids, ptexts, "clip_prompts", checkpoint, revision)
            ppath = cache_dir / f"clip_prompts_{pkey}.npz"
            if not ppath.exists():
                if model is None:
                    model, tokenizer = model_and_tokenizer(encoder, checkpoint, revision, selected_device)
                    actual_revision = getattr(model.config, "_commit_hash", None)
                pv, _ = encode_batch(ptexts, encoder=encoder, model=model,
                                     tokenizer=tokenizer, device=selected_device)
                save_npz(ppath, ids=pids, vectors=pv)
            with np.load(ppath, allow_pickle=False) as pf:
                if pf["ids"].astype(str).tolist() != pids:
                    raise ValueError("Incorrect cached CLIP prompt IDs")
                prompt_vectors = pf["vectors"].astype("float32")
            save_npz(output.parent / "clip_prompts.npz", ids=pids, vectors=prompt_vectors)
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    merged = np.vstack(arrays)
    sentiment = np.vstack(scores) if scores else None
    save_npz(output, ids=ids, vectors=merged, sentiment=sentiment)
    meta = {"encoder": encoder, "checkpoint": checkpoint, "requested_revision": revision,
            "resolved_commit_if_loaded": actual_revision,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "device": selected_device, "batch_size": batch_size,
            "shard_size": shard_size, "headline_count": len(ids), "dimension": merged.shape[1],
            "cache_key": key, "frozen": True,
            "notes": "FinBERT sentiment order: positive, negative, neutral. Hidden representation: last-layer CLS.",
            "clip_prompt_ids": pids if encoder == "clip" else []}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print("Saved:", output, "shape", merged.shape)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--encoder", required=True, choices=sorted(CHECKPOINT))
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", default=".cache/news_fusion")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    encode_manifest(manifest, encoder=args.encoder, output=Path(args.output),
                    cache_dir=Path(args.cache_dir), batch_size=args.batch_size,
                    shard_size=args.shard_size, revision=args.revision, device=args.device)

if __name__ == "__main__":
    main()
