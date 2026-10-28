"""Edit existing SD1.5 images with VAREdit-8B-512; no generation or training.

Run in the prepared varedit environment on a GPU node:
    python -u varedit_experiment.py

Edit SOURCE_RUN and CASES below first. Inspect each original image and write
its correction instruction yourself. Add more tuples to process more images.
The official VAREdit repository is required at REPO; no DFAR helpers are used.
Outputs: edited PNG, before/after PNG, and settings JSON for each case.
Re-running the same case overwrites its experiment outputs, not the baseline.
This is an exploratory editing test, not a GenEval evaluation or training run.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

# 1. Configuration: change these paths and instructions as needed.
ROOT = Path("/nfshomes/vyomwal5/DFAR")
REPO = ROOT / "external/VAREdit"
SOURCE_RUN = Path("/fs/nexus-scratch/vyomwal5/dfar_results/runs/baseline/sd15")
OUT = Path("/fs/nexus-scratch/vyomwal5/dfar_results/varedit/manual_sd15")
CFG, TAU, SEED = 3.0, 0.1, 42  # Guidance, sampling temperature, randomness seed.
CASES = [
    (405,0, "Elephant should be added to the scene, it should be placed on the right of the zebra."),
    (401,0, "Ball should be on the left of the bed, not on it."),
    (406,2,"Add a bear to the scene, then place it left of the bench"),
    (410,1,"photo of the cow is in the tv, it should be above cow"),
    (412,0,"Add a banana to the scene, it should be placed on the right of the suitcase."),  # (prompt_id, sample_id, instruction): fill after viewing image.
    # For a verified correct image, try: "Keep this image unchanged."
    # For a real color error, name the object and the required color explicitly.
]

if not CASES or any(not instruction.strip() for _, _, instruction in CASES):
    raise SystemExit("Fill CASES with image IDs and instructions before running.")

# Set cache BEFORE importing Hugging Face. Models download to scratch.
os.environ["HF_HOME"] = "/fs/nexus-scratch/vyomwal5/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import torch
assert torch.cuda.device_count() >= 2, "This version requires two GPUs."
torch.cuda.set_device(0)
import matplotlib
matplotlib.use("Agg")  # Save figures without needing a GUI on the cluster.
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import snapshot_download
from infer import load_model, generate_image

# 2. Download only the chosen AR checkpoint, visual tokenizer, and text encoder.
weights = Path(snapshot_download(
    "HiDream-ai/VAREdit",
    allow_patterns=["8B-512.pth", "infinity_vae_d56_f8_14_patchify.pth", "flan-t5-xl/*"],
))
print("Loading VAREdit on", torch.cuda.get_device_name(0), flush=True)
components = load_model(str(weights), str(weights / "8B-512.pth"), "8B", 512)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

# 3. Reuse the loaded model for all cases. No gradients or weight updates.
for prompt_id, sample_id, instruction in CASES:
    name = f"{prompt_id:05d}_{sample_id:04d}"
    folder = SOURCE_RUN / f"{prompt_id:05d}"
    source = folder / "samples" / f"{sample_id:04d}.png"
    metadata = json.loads((folder / "metadata.jsonl").read_text())
    prompt = metadata["prompt"]
    with Image.open(source) as image:
        original = image.convert("RGB")
    print(f"\n{name}\nPrompt: {prompt}\nEdit: {instruction}", flush=True)

    # Source RGB + instruction -> coarse-to-fine AR image tokens -> edited RGB.
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        edited = generate_image(components, str(source), instruction,
                                cfg=CFG, tau=TAU, seed=SEED)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    target = OUT / "after" / f"{prompt_id:05d}"
    (target / "samples").mkdir(parents=True, exist_ok=True)

    edited.save(target / "samples" / f"{sample_id:04d}.png")
    (target / "metadata.jsonl").write_text(json.dumps(metadata) + "\n")

    # 4. Save a comparison to open in VS Code or download from the cluster.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, image, title in zip(axes, [original, edited], ["Original", "Edited"]):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_comparison.png", dpi=150)
    plt.close(fig)

    # Record the actual settings beside each image; manual review stays separate.
    record = dict(source=str(source), prompt=prompt, instruction=instruction,
                  model="HiDream-ai/VAREdit", checkpoint="8B-512.pth",
                  weights_snapshot=weights.name, code_commit=commit,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  cfg=CFG, tau=TAU, seed=SEED, seconds=seconds)
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=2))
    print(f"Saved {name}: {seconds:.2f}s; results in {OUT}", flush=True)

# Review: did the edit work, did unrelated content survive, and is quality good?
# First-call timing may include initialization; it is not a formal benchmark.
# An unchanged instruction is an editing control, not tokenizer reconstruction.
