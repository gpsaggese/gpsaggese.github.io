"""
CLIP_ViT_Large_Task22_utils.py

Reusable utilities + a lightweight wrapper around the native `open_clip` API
for MVSA-style multimodal sentiment tasks.

Design goals
- Keep notebooks minimal (logic lives here).
- Make paths explicit and relative to the repo root.
- Provide a small wrapper layer over open_clip’s native API calls.
- Be robust to missing/corrupt images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd


open_clip = None
Image = None
UnidentifiedImageError = Exception

import torch  

try:
    import open_clip  
except Exception:
    open_clip = None

try:
    from PIL import Image, UnidentifiedImageError 
except Exception:
    Image = None
    UnidentifiedImageError = Exception

VALID_LABELS = {"positive", "neutral", "negative"}



# API surface

@dataclass(frozen=True)
class LabelBuildConfig:
    valid_labels: Tuple[str, ...] = ("positive", "neutral", "negative")
    default_label: str = "neutral"
    require_modal_agreement: bool = True


@dataclass(frozen=True)
class ClipModelSpec:
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    device: str = "auto"      
    precision: str = "fp32"   


@dataclass(frozen=True)
class EmbeddingConfig:
    normalize: bool = True
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class DatasetPaths:
    repo_root: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "DatasetPaths":
        repo_root = repo_root.resolve()
        return DatasetPaths(
            repo_root=repo_root,
            raw_dir=repo_root / "data" / "raw",
            processed_dir=repo_root / "data" / "processed",
            artifacts_dir=repo_root / "artifacts",
        )


def resolve_mvsa_paths(repo_root: Path) -> Dict[str, Path]:
    p = DatasetPaths.from_repo_root(repo_root)
    return {
        "repo_root": p.repo_root,
        "labels_txt": p.raw_dir / "labelResultAll.txt",
        "mvsa_root": p.raw_dir / "MVSA",
        "cleaned_labels_csv": p.processed_dir / "cleaned_labels.csv",
        "embeddings_parquet": p.artifacts_dir / "clip" / "mvsa_vitl14_img_txt_embeddings.parquet",
        "errors_csv": p.artifacts_dir / "clip" / "mvsa_clip_embedding_errors.csv",
    }



# Labels: native MVSA format -> clean CSV

def parse_mvsa_label_file(labels_txt: Path) -> pd.DataFrame:
    labels_txt = Path(labels_txt)
    if not labels_txt.exists():
        raise FileNotFoundError(f"Label file not found: {labels_txt}")

    # MVSA uses whitespace-separated columns: id ann1 ann2 ann3
    df = pd.read_csv(
        labels_txt,
        sep=r"\s+",
        header=None,
        names=["id", "ann1", "ann2", "ann3"],
        engine="python",
    )
    return df


def _majority_vote(labels: Sequence[str], cfg: LabelBuildConfig) -> str:
    valid = set(cfg.valid_labels)
    labels = [str(l).strip().lower() for l in labels if str(l).strip().lower() in valid]
    if not labels:
        return cfg.default_label
    counts = Counter(labels)
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    return sorted(winners)[0]  


def build_majority_labels(
    labels_df: pd.DataFrame,
    out_csv: Path,
    cfg: LabelBuildConfig = LabelBuildConfig(),
) -> pd.DataFrame:
    required = {"id", "ann1", "ann2", "ann3"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels_df missing columns: {sorted(missing)}")

    def parse_row(row) -> Dict[str, object]:
        text_labels: List[str] = []
        image_labels: List[str] = []
        for col in ["ann1", "ann2", "ann3"]:
            cell = str(row[col])
            if "," not in cell:
                continue
            a, b = [x.strip().lower() for x in cell.split(",", 1)]
            text_labels.append(a)
            image_labels.append(b)
        return {
            "id": row["id"],
            "text_majority": _majority_vote(text_labels, cfg),
            "image_majority": _majority_vote(image_labels, cfg),
        }

    out_df = labels_df.apply(parse_row, axis=1, result_type="expand")

    out_df["id"] = pd.to_numeric(out_df["id"], errors="coerce")
    out_df = out_df.dropna(subset=["id"]).copy()
    out_df["id"] = out_df["id"].astype(int)

    if cfg.require_modal_agreement:
        out_df = out_df[out_df["text_majority"] == out_df["image_majority"]].copy()

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df.reset_index(drop=True)


def load_cleaned_labels(cleaned_labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cleaned_labels_csv)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    for c in ["text_majority", "image_majority"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    df = df[df["text_majority"] == df["image_majority"]].copy()
    df = df[df["text_majority"].isin(VALID_LABELS)].copy()

    out = df.rename(columns={"text_majority": "label"})[["id", "label"]]
    return out.drop_duplicates("id").sort_values("id").reset_index(drop=True)


# Wrapper around open_clip API

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


class ClipEmbedder:
    """
    Lightweight wrapper around open_clip.

    Native API used:
      - open_clip.create_model_and_transforms(...)
      - open_clip.get_tokenizer(...)
      - model.encode_image(...)
      - model.encode_text(...)

    Wrapper responsibilities:
      - device selection and precision
      - normalization
      - robust handling of missing/corrupt images
    """

    def __init__(self, spec: ClipModelSpec, cfg: EmbeddingConfig = EmbeddingConfig()):
        if torch is None or open_clip is None or Image is None:
            raise ImportError("Missing deps: install torch, open_clip_torch, and pillow (PIL).")
        self.spec = spec
        self.cfg = cfg

        if spec.device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = spec.device
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.model_name,
            pretrained=spec.pretrained,
            device=self.device,
        )
        model.eval()

        if spec.precision == "fp16":
            model = model.half()

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(spec.model_name)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if torch is None:
            raise ImportError("torch is not installed or failed to import.")
        with torch.inference_mode():
            tokens = self.tokenizer(list(texts)).to(self.device)
            emb = self.model.encode_text(tokens).detach().float().cpu().numpy()
        if self.cfg.normalize:
            emb = _l2_normalize(emb)
        return emb

    def encode_images(self, image_paths: Sequence[Path]) -> Tuple[np.ndarray, List[Optional[str]]]:
        if torch is None:
            raise ImportError("torch is not installed or failed to import.")
        embs: List[np.ndarray] = []
        errors: List[Optional[str]] = []

        with torch.inference_mode():
            for p in image_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    x = self.preprocess(img).unsqueeze(0).to(self.device)
                    if self.spec.precision == "fp16":
                        x = x.half()
                    e = self.model.encode_image(x).detach().float().cpu().numpy()[0]
                    embs.append(e)
                    errors.append(None)
                except (FileNotFoundError, UnidentifiedImageError, OSError) as ex:
                    dim = 768
                    try:
                        dim = int(self.model.text_projection.shape[0])  
                    except Exception:
                        pass
                    embs.append(np.zeros((dim,), dtype=np.float32))
                    errors.append(f"{type(ex).__name__}: {ex}")

        emb = np.vstack(embs).astype(np.float32)
        if self.cfg.normalize:
            emb = _l2_normalize(emb)
        return emb, errors


# Training wrapper for TwoTowerClassifier_v2


def train_two_tower_classifier_v2(
    embeddings_parquet: Path,
    d_model: int = 256,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    modality_dropout_p: float = 0.10,
    dropout: float = 0.15,
    head_dropout: float = 0.20,
    device: str = "auto",
    use_class_weights: bool = True,
    label_smoothing: float = 0.05,
    patience: int = 8
):

    model_cfg = FusionConfig_v2(
        num_classes=3,
        input_dim=_infer_clip_dim_from_parquet(embeddings_parquet),
        d_model=d_model,
        modality_dropout_p=modality_dropout_p,
        dropout=dropout,
        head_dropout=head_dropout,
    )

    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        use_class_weights=use_class_weights,
        label_smoothing=label_smoothing,
    )

    return train_two_tower_v2(str(embeddings_parquet), model_cfg=model_cfg, train_cfg=train_cfg, patience=patience)





def _infer_clip_dim_from_parquet(embeddings_parquet: Path) -> int:
    df = pd.read_parquet(embeddings_parquet, columns=None)
    img_cols = [c for c in df.columns if c.startswith("img_")]
    if not img_cols:
        return 768
    # img_0..img_(D-1)
    return len(img_cols)

# Inference helpers (raw -> CLIP -> classifier)

def build_default_embedder(device: str = "auto") -> ClipEmbedder:
    spec = ClipModelSpec(device=device, precision="fp32")
    return ClipEmbedder(spec=spec, cfg=EmbeddingConfig(normalize=True))


def predict_sentiment_from_raw(
    img_path: str | Path,
    text: str,
    *,
    device: str = "auto",
    embedder: ClipEmbedder | None = None,
    model_path: str | Path | None = None,
) -> Dict[str, object]:
    import torch
    import numpy as np

    if embedder is None:
        embedder = build_default_embedder(device=device)

    # pick device for classifier
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = device

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

    # label mapping 
    id2label = ckpt.get("id2label", {0: "negative", 1: "neutral", 2: "positive"})
    label_order = [id2label[i] for i in range(len(id2label))]

    model_cfg = ckpt.get("model_cfg", None)
    if isinstance(model_cfg, dict):
        model_cfg = FusionConfig_v2(**model_cfg)
    elif model_cfg is None:
        model_cfg = FusionConfig_v2()

    model = TwoTowerClassifier_v2(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(dev)
    model.eval()

    # CLIP embeddings
    img_emb, errs = embedder.encode_images([Path(img_path)])
    txt_emb = embedder.encode_texts([text])
    if errs[0] is not None:
        raise FileNotFoundError(f"Image problem: {errs[0]}")

    x_img = torch.from_numpy(img_emb).float().to(dev)
    x_txt = torch.from_numpy(txt_emb).float().to(dev)

    with torch.no_grad():
        logits = model(x_img, x_txt)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = ["negative", "neutral", "positive"][pred_id]

    return {
        "pred_id": pred_id,
        "pred_label": pred_label,
        "probs": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        },
    }




# Embedding creation

from tqdm import tqdm


@dataclass(frozen=True)
class ClipConfig:
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    batch_size: int = 32
    max_items: Optional[int] = None  


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_cleaned_labels(cleaned_labels_csv: Path) -> pd.DataFrame:
    """
    Expect columns: id, text_majority, image_majority
    - drops junk rows like id == "ID"
    - keeps only rows where text_majority == image_majority
    - keeps only VALID_LABELS
    Returns df with columns [id, label] where id is int.
    """
    df = pd.read_csv(cleaned_labels_csv)

    required = {"id", "text_majority", "image_majority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cleaned_labels_csv} missing columns: {sorted(missing)}")

    # normalize types
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["text_majority"] = df["text_majority"].astype(str).str.strip().str.lower()
    df["image_majority"] = df["image_majority"].astype(str).str.strip().str.lower()

    # drop rows with bad ids like 'ID'
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    # enforce agreement and  valid labels
    df = df[df["text_majority"] == df["image_majority"]].copy()
    df = df[df["text_majority"].isin(VALID_LABELS)].copy()

    out = df.rename(columns={"text_majority": "label"})[["id", "label"]]
    out = out.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    return out


def _find_image(raw_data_path: Path, sample_id: int) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = raw_data_path / f"{sample_id}{ext}"
        if p.exists():
            return p
    return None


def _load_text(raw_data_path: Path, sample_id: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Prefer raw_data_path/clean_text/<id>.txt if it exists, else raw_data_path/<id>.txt
    """
    clean_path = raw_data_path / "clean_text" / f"{sample_id}.txt"
    raw_path = raw_data_path / f"{sample_id}.txt"
    path = clean_path if clean_path.exists() else raw_path

    if not path.exists():
        return None, "missing_text"

    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None, "text_read_error"

    if not txt:
        return None, "empty_text"

    return txt, None


def _load_image(raw_data_path: Path, sample_id: int) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Returns (PIL.Image RGB, None) or (None, error_code).
    """
    img_path = _find_image(raw_data_path, sample_id)
    if img_path is None:
        return None, "missing_image"

    try:
        if img_path.stat().st_size == 0:
            return None, "empty_image"
        with Image.open(img_path) as img:
            img.load()
            return img.convert("RGB"), None
    except UnidentifiedImageError:
        return None, "bad_image_format"
    except Exception:
        return None, "image_read_error"


def create_embeddings(
    CLEANED_LABELS_CSV: str | Path,
    RAW_DATA_PATH: str | Path,
    *,
    artifacts_root: str | Path = "artifacts",
    output_name: str = "mvsa_vitl14_img_txt_embeddings.parquet",
    error_log_name: str = "mvsa_embedding_errors.csv",
    config: ClipConfig = ClipConfig(),
) -> Path:
    """
    Create CLIP embeddings for MVSA image+text pairs.

    Inputs:
      - CLEANED_LABELS_CSV: path to cleaned_labels.csv (id, text_majority, image_majority)
      - RAW_DATA_PATH: folder containing images + <id>.txt 

    Outputs (under artifacts_root/clip/):
      - Parquet with columns: id, text, label, img_0.., txt_0..
      - CSV error log 

    Returns:
      - Path to the written parquet file.
    """
    cleaned_labels_csv = Path(CLEANED_LABELS_CSV).expanduser().resolve()
    raw_data_path = Path(RAW_DATA_PATH).expanduser().resolve()

    if not cleaned_labels_csv.exists():
        raise FileNotFoundError(f"Missing cleaned labels CSV: {cleaned_labels_csv}")
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Missing RAW_DATA_PATH directory: {raw_data_path}")

    artifacts_dir = Path(artifacts_root).expanduser().resolve() / "clip"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_file = artifacts_dir / output_name
    err_file = artifacts_dir / error_log_name

    labels_df = _load_cleaned_labels(cleaned_labels_csv)
    if config.max_items is not None:
        labels_df = labels_df.head(config.max_items).reset_index(drop=True)

    device = _get_device()
    use_autocast = device in {"cuda", "mps"}

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=config.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(config.model_name)
    model = model.to(device).eval()

    # Storage
    meta: List[Tuple[int, str, str]] = []
    img_feats: List[np.ndarray] = []
    txt_feats: List[np.ndarray] = []

    # Errors
    err_rows: List[Dict[str, object]] = []
    err_counts = Counter()

    batch_imgs: List[torch.Tensor] = []
    batch_texts: List[str] = []
    batch_ids: List[int] = []
    batch_labels: List[str] = []

    def flush() -> None:
        if not batch_ids:
            return

        imgs = torch.stack(batch_imgs, dim=0).to(device)
        toks = tokenizer(batch_texts).to(device)

        with torch.no_grad():
            if use_autocast:
                dtype = torch.float16
                device_type = "cuda" if device == "cuda" else "mps"
                with torch.autocast(device_type=device_type, dtype=dtype):
                    im = model.encode_image(imgs)
                    tx = model.encode_text(toks)
            else:
                im = model.encode_image(imgs)
                tx = model.encode_text(toks)

        # L2 normalize
        im = im / im.norm(dim=-1, keepdim=True)
        tx = tx / tx.norm(dim=-1, keepdim=True)

        im_np = im.detach().cpu().numpy().astype("float32")
        tx_np = tx.detach().cpu().numpy().astype("float32")

        for i in range(len(batch_ids)):
            meta.append((batch_ids[i], batch_texts[i], batch_labels[i]))
            img_feats.append(im_np[i])
            txt_feats.append(tx_np[i])

        batch_imgs.clear()
        batch_texts.clear()
        batch_ids.clear()
        batch_labels.clear()

    for row in tqdm(labels_df.itertuples(index=False), total=len(labels_df), desc="CLIP encoding"):
        sample_id = int(row.id)
        label = str(row.label)

        text, text_err = _load_text(raw_data_path, sample_id)
        if text_err:
            err_counts[text_err] += 1
            err_rows.append({"id": sample_id, "type": text_err})
            continue

        img, img_err = _load_image(raw_data_path, sample_id)
        if img_err:
            err_counts[img_err] += 1
            err_rows.append({"id": sample_id, "type": img_err})
            continue

        batch_imgs.append(preprocess(img))
        batch_texts.append(text)
        batch_ids.append(sample_id)
        batch_labels.append(label)

        if len(batch_ids) >= config.batch_size:
            flush()

    flush()

    if not meta:
        raise RuntimeError(
            "No valid (image, text) pairs were encoded. "
            "Check RAW_DATA_PATH and IDs in cleaned_labels.csv."
        )

    meta_df = pd.DataFrame(meta, columns=["id", "text", "label"])

    emb_img = np.stack(img_feats)  
    emb_txt = np.stack(txt_feats)  

    df_img = pd.DataFrame(emb_img, columns=[f"img_{i}" for i in range(emb_img.shape[1])])
    df_txt = pd.DataFrame(emb_txt, columns=[f"txt_{i}" for i in range(emb_txt.shape[1])])

    out_df = pd.concat([meta_df.reset_index(drop=True), df_img, df_txt], axis=1)
    out_df.to_parquet(out_file, index=False)

    if err_rows:
        pd.DataFrame(err_rows).sort_values("id").to_csv(err_file, index=False)

    print(f"Saved embeddings: {out_file}  (rows={len(out_df)})")
    if err_rows:
        print(f"Saved error log:  {err_file}  (summary={dict(err_counts)})")

    return out_file

# Model

import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class FusionConfig_v2:
    num_classes: int = 3
    input_dim: int = 768           
    d_model: int = 256             
    dropout: float = 0.15
    head_dropout: float = 0.20
    use_layernorm: bool = True
    l2_normalize_inputs: bool = True

    # randomly drop a modality during training
    modality_dropout_p: float = 0.10   


class Projector(nn.Module):
    """Linear -> GELU -> Dropout -> Linear (+ optional LN)."""
    def __init__(self, in_dim: int, d_model: int, dropout: float, use_layernorm: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(x))


class ResidualMLP(nn.Module):
    """2-layer MLP with residual + LN (stabilizes training)."""
    def __init__(self, dim: int, dropout: float, use_layernorm: bool):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.fc1(x)))
        h = self.drop(self.fc2(h))
        return self.ln(x + h)


class TwoTowerClassifier_v2(nn.Module):
    """
    Likely-better fusion for (image_emb, text_emb) vector inputs:

    1) Optional L2 norm on inputs
    2) Project each modality to d_model
    3) Feature-wise gate: g in [0,1]^d decides mix per dimension
    4) Interaction features: prod, abs diff
    5) Strong head with residual MLP blocks
    """
    def __init__(self, cfg: FusionConfig_v2):
        super().__init__()
        self.cfg = cfg

        self.img_proj = Projector(cfg.input_dim, cfg.d_model, cfg.dropout, cfg.use_layernorm)
        self.txt_proj = Projector(cfg.input_dim, cfg.d_model, cfg.dropout, cfg.use_layernorm)

        
        self.gate = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        fused_dim = 4 * cfg.d_model

        self.fuse_ln = nn.LayerNorm(fused_dim) if cfg.use_layernorm else nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
        )

        self.res1 = ResidualMLP(fused_dim, cfg.head_dropout, cfg.use_layernorm)
        self.res2 = ResidualMLP(fused_dim, cfg.head_dropout, cfg.use_layernorm)

        self.classifier = nn.Linear(fused_dim, cfg.num_classes)

    def _maybe_modality_dropout(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.cfg.modality_dropout_p
        if (not self.training) or p <= 0.0:
            return img, txt

        # With prob p/2 drop image, with prob p/2 drop text 
        r = torch.rand(1, device=img.device).item()
        if r < p / 2:
            img = torch.zeros_like(img)
        elif r < p:
            txt = torch.zeros_like(txt)
        return img, txt

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if self.cfg.l2_normalize_inputs:
            image_emb = F.normalize(image_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)

        image_emb, text_emb = self._maybe_modality_dropout(image_emb, text_emb)

        zi = self.img_proj(image_emb) 
        zt = self.txt_proj(text_emb)   

        # feature-wise gate
        g = torch.sigmoid(self.gate(torch.cat([zi, zt], dim=-1)))  
        mix = g * zi + (1.0 - g) * zt                              

        prod = zi * zt                                             
        diff = torch.abs(zi - zt)                                 

        fused = torch.cat([mix, zi, zt, prod + diff], dim=-1)       
        fused = self.fuse_ln(fused)

        h = self.head(fused)
        h = self.res1(h)
        h = self.res2(h)

        logits = self.classifier(h)
        return logits

    @torch.no_grad()
    def predict_proba(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(image_emb, text_emb), dim=-1)
    


# Trining loop


from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    seed: int = 7
    device: str = "auto"  
    val_split: float = 0.2

    # extras
    use_amp: bool = True              
    grad_clip_norm: float = 1.0       
    label_smoothing: float = 0.05     
    use_class_weights: bool = True   
    num_workers: int = 0


class EmbeddingDataset(Dataset):
    def __init__(self, img: np.ndarray, txt: np.ndarray, y: np.ndarray):
        self.img = torch.from_numpy(img).float()
        self.txt = torch.from_numpy(txt).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.img[idx], self.txt[idx], self.y[idx]


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_xy_from_parquet(parquet_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)

    img_cols = [c for c in df.columns if c.startswith("img_")]
    txt_cols = [c for c in df.columns if c.startswith("txt_")]
    if not img_cols or not txt_cols:
        raise ValueError("Could not find img_* or txt_* columns. Did you generate embeddings parquet?")

    X_img = df[img_cols].to_numpy(dtype=np.float32)
    X_txt = df[txt_cols].to_numpy(dtype=np.float32)

    labels = df["label"].astype(str).str.lower().to_list()
    y = np.array([LABEL2ID.get(l, 1) for l in labels], dtype=np.int64)  
    return X_img, X_txt, y


def _make_class_weights(y_tr: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights).float()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, ce: nn.Module) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    for img, txt, y in loader:
        img, txt, y = img.to(device), txt.to(device), y.to(device)
        logits = model(img, txt)
        loss = ce(logits, y)

        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.size(0))

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def predict_labels(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []

    for img, txt, y in loader:
        img, txt = img.to(device), txt.to(device)
        logits = model(img, txt)
        pred = logits.argmax(dim=1)

        y_true.append(y.numpy())
        y_pred.append(pred.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def train_two_tower_v2(
    parquet_path: str,
    model_cfg: FusionConfig_v2 = FusionConfig_v2(),
    train_cfg: TrainConfig = TrainConfig(),
    patience: int = 8,
    min_delta: float = 1e-4,
    artifacts_root: str | Path = "artifacts",
    ckpt_name: str = "two_tower_v2_best.pt",
) -> Dict[str, object]:
    """
    Trains TwoTowerClassifier_v2 and saves the best checkpoint (by val_macro_f1)
    """
    _set_seed(train_cfg.seed)
    device = _pick_device(train_cfg.device)

    X_img, X_txt, y = load_xy_from_parquet(parquet_path)

    # stratified split
    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=train_cfg.val_split,
        random_state=train_cfg.seed,
        stratify=y,
    )

    ds_tr = EmbeddingDataset(X_img[tr_idx], X_txt[tr_idx], y[tr_idx])
    ds_va = EmbeddingDataset(X_img[val_idx], X_txt[val_idx], y[val_idx])

    dl_tr = DataLoader(
        ds_tr,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = TwoTowerClassifier_v2(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # Scheduler on VAL LOSS 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

    class_weights = None
    if train_cfg.use_class_weights:
        class_weights = _make_class_weights(y[tr_idx], num_classes=model_cfg.num_classes).to(device)

    ce = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=train_cfg.label_smoothing,
    )
    print("use_class_weights =", train_cfg.use_class_weights, "class_weights =", class_weights)

    use_amp = (train_cfg.use_amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict[str, float]] = []

    # Early stopping and checkpointing on VAL MACRO F1 (maximize)
    best_state = None
    patience_ctr = 0

    best_epoch = 0
    best_val_loss = float("inf")
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_val_macro_f1 = -1.0

    for ep in range(1, train_cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for img, txt, yy in dl_tr:
            img, txt, yy = img.to(device), txt.to(device), yy.to(device)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(img, txt)
                    loss = ce(logits, yy)

                scaler.scale(loss).backward()

                if train_cfg.grad_clip_norm and train_cfg.grad_clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

                scaler.step(opt)
                scaler.update()
            else:
                logits = model(img, txt)
                loss = ce(logits, yy)
                loss.backward()

                if train_cfg.grad_clip_norm and train_cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

                opt.step()

            loss_sum += float(loss.item()) * yy.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yy).sum().item())
            total += int(yy.size(0))

        train_metrics = {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}
        val_metrics = evaluate(model, dl_va, device=device, ce=ce)

        # LR schedule based on val loss (smooth)
        scheduler.step(val_metrics["loss"])
        lr_now = opt.param_groups[0]["lr"]

        row = {
            "epoch": ep,
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            "lr": float(lr_now),
        }
        history.append(row)

        print(
            f"[ep {ep:02d}] "
            f"lr={lr_now:.2e} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.3f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f} "
            f"val_macro_f1={row['val_macro_f1']:.3f}"
        )

        # Early stopping on macro_f1 max
        score = float(val_metrics["macro_f1"])

        if score > best_val_macro_f1 + min_delta:
            best_val_macro_f1 = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0

            best_epoch = ep
            best_val_loss = float(val_metrics["loss"])  
            best_train_acc = float(train_metrics["acc"])
            best_val_acc = float(val_metrics["acc"])
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(
                    f"Early stopping at epoch {ep} "
                    f"(best_val_macro_f1={best_val_macro_f1:.4f} @ epoch {best_epoch})"
                )
                break

    # restoring best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Per-class metrics on best checkpoint
    y_true, y_pred = predict_labels(model, dl_va, device)
    names = [ID2LABEL[i] for i in range(model_cfg.num_classes)]

    print("\nPer-class metrics (VAL @ best checkpoint):")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    print(
        f"[best_epoch {best_epoch:02d}] "
        f"best_val_macro_f1={best_val_macro_f1:.4f} "
        f"best_val_acc={best_val_acc:.4f} "
        f"best_val_loss={best_val_loss:.4f}"
    )

       
        # saving best model 
        
    artifacts_root = Path(artifacts_root).expanduser().resolve()
    ckpt_path = artifacts_root / "model" / ckpt_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe_cfg(x):
        if x is None:
            return None
        if is_dataclass(x):
            return asdict(x)
        if isinstance(x, dict):
            return x
        
        return str(x)

    # Save only safe / portable objects no custom class pickling
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best": {
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "train_acc": best_train_acc,
                "val_acc": best_val_acc,
                "val_macro_f1": best_val_macro_f1,
            },
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "model_cfg": _safe_cfg(model_cfg),   
            "train_cfg": _safe_cfg(train_cfg),   
        },
        ckpt_path,
    )
    print(f"Saved best checkpoint to: {ckpt_path}")

    return { "model": model, "history": pd.DataFrame(history), "device": device, "best": { "epoch": best_epoch, "val_loss": best_val_loss, "train_acc": best_train_acc, "val_acc": best_val_acc, "val_macro_f1": best_val_macro_f1, }}


# Plotting

import matplotlib.pyplot as plt

def plot_training_history_from_out(out, best_epoch=None):
    """
    Works with out["history"] being either:
      - a pandas DataFrame, or
      - a list of dicts
    and handles common column name variants.
    """
    hist = out.get("history", None)
    if hist is None:
        raise KeyError("out does not contain 'history'. Make sure train_two_tower_classifier_v2 returns it.")

    # normalizing to DataFrame
    try:
        import pandas as pd
        df = hist if hasattr(hist, "columns") else pd.DataFrame(hist)
    except Exception:
        raise ValueError("Could not convert history to a DataFrame. Inspect out['history'].")

   
    rename_map = {}
    if "val_macro_f1" not in df.columns and "val_f1" in df.columns:
        rename_map["val_f1"] = "val_macro_f1"
    if "epoch" not in df.columns and "epochs" in df.columns:
        rename_map["epochs"] = "epoch"
    df = df.rename(columns=rename_map)

    df = df.sort_values("epoch").reset_index(drop=True)

    # infer best epoch 
    if best_epoch is None:
        if "best_epoch" in out:
            best_epoch = out["best_epoch"]
        elif "val_macro_f1" in df.columns:
            best_epoch = int(df.loc[df["val_macro_f1"].idxmax(), "epoch"])
        elif "val_loss" in df.columns:
            best_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])

    # helper for vertical line
    def _mark_best():
        if best_epoch is not None:
            plt.axvline(best_epoch, linestyle="--", label=f"best_epoch={best_epoch}")

    # Loss 
    plt.figure()
    if "train_loss" in df.columns: plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:   plt.plot(df["epoch"], df["val_loss"],   label="val_loss")
    _mark_best()
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.show()

    # Accuracy 
    plt.figure()
    if "train_acc" in df.columns: plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    if "val_acc" in df.columns:   plt.plot(df["epoch"], df["val_acc"],   label="val_acc")
    _mark_best()
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch")
    plt.legend(); plt.show()

    #  Macro F1
    if "val_macro_f1" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["val_macro_f1"], label="val_macro_f1")
        _mark_best()
        plt.xlabel("Epoch"); plt.ylabel("Macro F1"); plt.title("Val Macro F1 vs Epoch")
        plt.legend(); plt.show()

    #  LR
    if "lr" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["lr"], label="lr")
        _mark_best()
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("Learning Rate vs Epoch")
        plt.legend(); plt.show()