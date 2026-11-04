# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description
# ### CLIP API: Encoding Energy News Headlines
#
# This notebook demonstrates how to use the frozen text encoder of
# `openai/clip-vit-large-patch14` to represent energy news headlines
# as numerical embeddings.
#
# The walkthrough covers dependency installation, model loading,
# text tokenization, and embedding extraction. These representations
# will later be used in the oil and natural gas forecasting example.
#
# The headlines below are illustrative examples used to verify the API.
# This notebook does not yet evaluate forecasting performance.
#
# ## Learning objectives
#
# - Load CLIP with `CLIPModel` and `CLIPProcessor`.
# - Freeze model parameters and run inference.
# - Convert headlines into 768-dimensional embeddings.
# - Check embedding shapes and numerical validity.
#
# ## References
#
# - [CLIP paper](https://arxiv.org/abs/2103.00020)
# - [Model card](https://huggingface.co/openai/clip-vit-large-patch14)
# - [Transformers CLIP documentation](https://huggingface.co/docs/transformers/v4.53.2/model_doc/clip)

# %% [markdown]
# ## Install dependencies

# %%
# Install the CPU version of PyTorch.
# %pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install CLIP support and the forecasting utilities.
# %pip install -q transformers==4.53.2 scikit-learn==1.7.1

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from transformers import CLIPModel, CLIPProcessor
from IPython.display import display

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("scikit-learn:", sklearn.__version__)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("CLIP and forecasting imports passed.")

# %% [markdown]
# ## Load the frozen model and encode sample headlines

# %%
from pathlib import Path

# Cache model files in the mounted project folder.
cache_dir = Path.cwd() / ".cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

# Load the assigned CLIP checkpoint.
model_id = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
model = CLIPModel.from_pretrained(model_id, cache_dir=str(cache_dir))

# Freeze all parameters and disable training behavior.
model.requires_grad_(False)
model.eval()

# Use illustrative headlines to verify text encoding.
headlines = [
    "OPEC announces a reduction in oil production.",
    "Oil demand weakens as economic activity slows.",
    "Cold weather increases demand for natural gas.",
]

inputs = processor(
    text=headlines,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)

# Extract one embedding per headline without computing gradients.
with torch.inference_mode():
    embeddings = model.get_text_features(**inputs)

assert embeddings.shape == (3, 768)
assert torch.isfinite(embeddings).all()
assert not any(parameter.requires_grad for parameter in model.parameters())

print("Model:", model_id)
print("Embedding shape:", tuple(embeddings.shape))
print("Frozen CLIP text encoding passed.")

# %%
from pathlib import Path

# Exclude downloaded models and generated Python/notebook caches.
ignore_path = Path.cwd() / ".gitignore"
existing = ignore_path.read_text() if ignore_path.exists() else ""
patterns = [".cache/", "__pycache__/", ".ipynb_checkpoints/"]
missing = [
    pattern for pattern in patterns if pattern not in existing.splitlines()
]

if missing:
    separator = "\n" if existing and not existing.endswith("\n") else ""
    ignore_path.write_text(existing + separator + "\n".join(missing) + "\n")

print("Project directory:", Path.cwd())
print("Cache exclusions saved.")

# %% [markdown]
# ## Compare headlines with energy-event prompts
#
# We compare headline embeddings with embeddings of predefined energy
# events using cosine similarity. Higher scores indicate greater
# similarity in CLIP's representation space.
#
# These scores are experimental text features, not event probabilities
# or predictions of price direction.

# %%
# Define illustrative energy-event prompts.
event_prompts = [
    "OPEC production cut",
    "weak oil demand",
    "increased natural gas demand due to cold weather",
    "oil supply disruption",
]

# Tokenize prompts using the same processor as the headlines.
prompt_inputs = processor(
    text=event_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)

# Encode prompts with the frozen model.
with torch.inference_mode():
    prompt_embeddings = model.get_text_features(**prompt_inputs)

# Normalize vectors so their dot products give cosine similarities.
headline_vectors = torch.nn.functional.normalize(embeddings, dim=-1)
prompt_vectors = torch.nn.functional.normalize(prompt_embeddings, dim=-1)
similarities = headline_vectors @ prompt_vectors.T

# Display one row per headline and one column per event prompt.
scores = pd.DataFrame(
    similarities.cpu().numpy(),
    index=pd.Index(headlines, name="Headline"),
    columns=event_prompts,
)

assert scores.shape == (3, 4)
assert np.isfinite(scores.to_numpy()).all()

display(scores.round(3))

# %% [markdown]
# ### Interpretation
#
# Each illustrative headline has its highest cosine similarity with the
# corresponding event prompt. This verifies that the encoding and similarity
# calculation work on these three examples.
#
# Scores for other prompts are also high, so absolute similarity values
# should not be interpreted as calibrated probabilities. These examples
# have closely matching wording and do not establish performance on real
# news or predictive value for future returns.
#
# The forecasting experiment will test whether these features improve
# predictions beyond historical price features on unseen dates.

# %%
