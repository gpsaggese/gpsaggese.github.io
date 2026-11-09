"""Local, pinned FinBERT tone inference. Scores are not return probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_ID = "Xenova/finbert"
MODEL_REVISION = "8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20"
MODEL_FILE = "onnx/model_quantized.onnx"


def cached_model_file(snapshot, filename):
    """Read only an explicit immutable local snapshot; never initiate a download."""
    snapshot = Path(snapshot)
    if snapshot.name != MODEL_REVISION or filename not in {MODEL_FILE, "tokenizer.json", "config.json"}:
        raise ValueError("Require the pinned FinBERT snapshot and known model files")
    candidate = snapshot / filename
    if not candidate.is_file():
        raise FileNotFoundError("Pinned local FinBERT file is absent: " + str(candidate))
    return candidate


class FinBertTone:
    """Run the ONNX conversion of ProsusAI/finbert without a remote API."""

    def __init__(self, *, offline: bool = False, cached_snapshot=None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        def fetch(filename):
            if cached_snapshot is not None:
                if not offline:
                    raise ValueError("An explicit cache snapshot requires offline=True")
                return cached_model_file(cached_snapshot, filename)
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION,
                                       token=False, local_files_only=offline))

        model_path, tokenizer_path, config_path = map(fetch, [MODEL_FILE, "tokenizer.json", "config.json"])
        config = json.loads(config_path.read_text())
        self.labels = {int(k): v.lower() for k, v in config["id2label"].items()}
        if set(self.labels.values()) != {"positive", "negative", "neutral"}:
            raise ValueError("Unexpected FinBERT label mapping")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.np = np
        self.metadata = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "weights": MODEL_FILE,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "runtime": "onnxruntime", "runtime_version": ort.__version__,
            "device": "CPU", "max_tokens": 512, "quantized": True,
            "trained_in_this_project": False, "score_meaning": "Uncalibrated text-tone softmax; not probability of a stock rising.",
        }

    def predict(self, texts: list[str]) -> list[dict]:
        np = self.np
        results = []
        for offset in range(0, len(texts), 8):
            batch = texts[offset:offset + 8]
            self.tokenizer.no_truncation()
            lengths = [len(self.tokenizer.encode(text).ids) for text in batch]
            self.tokenizer.enable_truncation(max_length=512)
            encoded = self.tokenizer.encode_batch(batch)
            candidates = {
                "input_ids": np.asarray([x.ids for x in encoded], dtype=np.int64),
                "attention_mask": np.asarray([x.attention_mask for x in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([x.type_ids for x in encoded], dtype=np.int64),
            }
            logits = self.session.run(None, {x.name: candidates[x.name] for x in self.session.get_inputs()})[0]
            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            for text, length, values in zip(batch, lengths, probabilities):
                scores = {self.labels[i]: float(value) for i, value in enumerate(values)}
                if not all(np.isfinite(value) for value in scores.values()):
                    raise ValueError("FinBERT returned non-finite probabilities")
                results.append({"status": "Available", "label": max(scores, key=scores.get),
                                "scores": scores, "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "input_tokens": length, "truncated": length > 512,
                                "interpretation": self.metadata["score_meaning"]})
        return results
