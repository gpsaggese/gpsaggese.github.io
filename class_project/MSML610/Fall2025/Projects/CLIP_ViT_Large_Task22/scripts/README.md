# CLIP embeddings for MVSA 

Use CLIP (OpenAI / OpenCLIP) to make image-text embeddings for the MVSA dataset, and then store them as Parquet files so they may be used in further modeling.

.
data/raw/mvsa/
   labelResultAll.txt
   data/              # 2501.jpg, 2502.txt, ...
artifacts/
     clip/            # outputs + logs land here
scripts/
   embed_clip.py
requirements.txt

# SETUP

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Data assumptions

Images and texts share the same folder and basename: 2501.jpg, 2501.txt.
Supported image extensions: .jpg, .jpeg, .png.
labelResultAll.txt has an ID and three columns of text_label,image_label pairs; a majority vote yields the final label.

# Run (default config in script)

# 2.5k subset (quick check)
python scripts/embed_clip.py

# Full run 
# If your script exposes a LIMIT or ID range, set it. Otherwise the script processes all ids found.
# Example flags if you added argparse:
# python scripts/embed_clip.py  limit 19600

**Device:** If available, uses the Apple M-series (MPS); if not, it uses the CPU or CUDA.  For 16 GB of RAM, the default batch size is safe; reduce it if you observe pressure.

**Results**

Artifacts/clip/msa_vitl14_img+txt-embeddings.parquet contains the embeddings.
Columns: id, text, label, txt_0..txt_{D-1}, img_0..img_{D-1}
D = 768 for ViT-L/14
Error log: artifacts/clip/embedding_errors.csv
Id, path, and kind (such as missing_image, empty_text, empty_image, decode_error) are the columns.
After a run, a summary is printed.
If artifacts/clip/ already exists, the script won't recreate it.  If you use the same filename for an output file, it will be overwritten.

# Advice
**Faster/smaller:** change the script to ViT-B-32.
**Stability:** maintain a partial checkpoint (optional in code) per N samples.
Ignore the **Pandas fragmentation warning** or write embeddings by concatenating arrays all at once (the most recent version has already been optimized).

**Troubleshooting**
"No such file or directory": Verify LABELS_TXT and DATA_DIR.
Verify that the names of many missing pairs (such as 2499.jpg & 2499.txt) match perfectly.
CPU slowness: verify MPS using:

import torch; print(torch.backends.mps.is_available())

**Parquet: What is it?**
For huge numerical tables, a columnar file format that loads with pandas is significantly faster and smaller than CSV. read_parquet.



