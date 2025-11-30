#!/usr/bin/env python
import os, re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import open_clip

# CONFIG 
DATA_DIR   = Path(os.path.expanduser("~/Downloads/MVSA/data"))
LABELS_TXT = os.path.expanduser("~/Downloads/MVSA/labelResultAll.txt")
OUT_FILE   = "artifacts/clip/msa_vitl14_img+txt-embeddings.parquet"   # artifacts/clip 
ERR_LOG    = "artifacts/clip/embedding_errors.csv"                    

MODEL_NAME = "ViT-L-14"        
PRETRAINED = "openai"
MAX_ITEMS  = 19600             
BATCH_SIZE = 8                 

# device 
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# model + transforms 
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(device).eval()
use_autocast = device in {"mps", "cuda"}

# helpers 
def parse_label_row(parts):
    # parts: [id, col1, col2, col3], where each col like "positive,neutral"
    valid = {"positive", "neutral", "negative"}
    id_str = parts[0].strip()
    raw_pairs = [p.strip() for p in parts[1:]]
    labels = []
    for pair in raw_pairs:
        if not pair:
            continue
        labels.extend([s.strip() for s in pair.split(",") if s.strip()])
    labels = [l for l in labels if l in valid]
    if not labels:
        majority = "neutral"
    else:
        counts = Counter(labels).most_common()
        top = counts[0][1]
        tied = [l for l, c in counts if c == top]
        if len(tied) == 1:
            majority = tied[0]
        else:
            first_text = raw_pairs[0].split(",")[0].strip()
            majority = first_text if first_text in valid else "neutral"
    return int(id_str), majority

def read_labels_table(labels_path):
    rows = []
    with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if parts[0].lower() == "id" or not parts[0].isdigit():
                continue
            parts = parts[:4] if len(parts) >= 4 else parts
            try:
                ex_id, lab = parse_label_row(parts)
                rows.append((ex_id, lab))
            except Exception:
                pass
    return pd.DataFrame(rows, columns=["id", "label"]).drop_duplicates("id")

def find_image_for_id(i: int):
    for ext in (".jpg", ".jpeg", ".png"):
        p = DATA_DIR / f"{i}{ext}"
        if p.exists():
            return p
    return None

def load_text_for_id(i: int):
    p = DATA_DIR / f"{i}.txt"
    if not p.exists():
        return None, "missing_text"
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, "text_read_error"
    if len(txt.strip()) == 0:
        return None, "empty_text"
    return txt.strip(), None

def load_image_for_id(i: int):
    p = find_image_for_id(i)
    if p is None:
        return None, "missing_image"
    try:
        if p.stat().st_size == 0:
            return None, "empty_image"
        img = Image.open(p)
        img.load()
        img = img.convert("RGB")
        return img, None
    except UnidentifiedImageError:
        return None, "bad_image_format"
    except Exception:
        return None, "image_read_error"

# main 
def main():
    labels_df = read_labels_table(LABELS_TXT)
    if MAX_ITEMS is not None:
        labels_df = labels_df.sort_values("id").head(MAX_ITEMS).reset_index(drop=True)

    records = []      
    img_feats = []    
    txt_feats = []    

    err_rows = []     
    err_counts = defaultdict(int)
    batch_imgs = []
    batch_txts = []
    batch_ids  = []
    batch_labs = []
    batch_text_raw = []

    def flush_batch():
        if not batch_ids:
            return
        imgs = torch.stack(batch_imgs, dim=0).to(device)
        toks = tokenizer(batch_text_raw).to(device)

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type=("mps" if device == "mps" else "cuda"), dtype=torch.float16):
                    im = model.encode_image(imgs)
                    tx = model.encode_text(toks)
            else:
                im = model.encode_image(imgs)
                tx = model.encode_text(toks)

        im = im / im.norm(dim=-1, keepdim=True)
        tx = tx / tx.norm(dim=-1, keepdim=True)

        im_np = im.cpu().numpy()
        tx_np = tx.cpu().numpy()

        for k in range(len(batch_ids)):
            img_feats.append(im_np[k])
            txt_feats.append(tx_np[k])
            records.append((batch_ids[k], batch_text_raw[k], batch_labs[k]))

        batch_imgs.clear()
        batch_txts.clear()
        batch_ids.clear()
        batch_labs.clear()
        batch_text_raw.clear()

    pbar = tqdm(labels_df.itertuples(index=False), total=len(labels_df), desc="Encoding")
    for row in pbar:
        i = int(row.id); lab = row.label

        txt, txt_err = load_text_for_id(i)
        if txt_err:
            err_counts[txt_err] += 1
            err_rows.append({"id": i, "type": txt_err})
            continue

        img, img_err = load_image_for_id(i)
        if img_err:
            err_counts[img_err] += 1
            err_rows.append({"id": i, "type": img_err})
            continue

        img_t = preprocess(img)
        batch_imgs.append(img_t)
        batch_txts.append(txt)   # keep raw; tokenize in batch
        batch_ids.append(i)
        batch_labs.append(lab)
        batch_text_raw.append(txt)

        if len(batch_ids) == BATCH_SIZE:
            flush_batch()

    flush_batch()

    if len(records) == 0:
        raise RuntimeError("No valid (image, text) pairs were encoded. Check DATA_DIR and file names.")

    meta_df = pd.DataFrame(records, columns=["id", "text", "label"])
    emb_img = np.stack(img_feats).astype("float32")
    emb_txt = np.stack(txt_feats).astype("float32")

    # building once to avoid fragmentation warnings
    df_img = pd.DataFrame(emb_img, columns=[f"img_{k}" for k in range(emb_img.shape[1])])
    df_txt = pd.DataFrame(emb_txt, columns=[f"txt_{k}" for k in range(emb_txt.shape[1])])
    df = pd.concat([meta_df.reset_index(drop=True), df_img, df_txt], axis=1)

    
    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUT_FILE}")

    if err_rows:
        err_df = pd.DataFrame(err_rows).sort_values("id")
        err_df.to_csv(ERR_LOG, index=False)
        print(f"Error summary: {dict(err_counts)}")
        print(f"Detailed log written to {ERR_LOG}")
    else:
        print("No file errors encountered.")

if __name__ == "__main__":
    main()
