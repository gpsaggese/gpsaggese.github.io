# Learn CLIP in 60 Minutes: Multimodal Sentiment Analysis

MSML610 Fall 2026, task #593. Team: Eunsoo Suk, Hardik Kamboj, Nishanth Reddy
Adidela.

This project is a tutorial on **CLIP** (`openai/clip-vit-large-patch14`), a
vision-language model that maps images and texts into a shared 768-d space.
We apply it to **sentiment analysis of tweets with an image** (the
MVSA-Multiple dataset): predict whether a post is negative, neutral, or
positive.

CLIP stays frozen throughout. We compare:
- **Baselines**: majority class, zero-shot CLIP (prompts such as "a photo with
  a positive mood", no training), and logistic regression on the frozen
  embeddings.
- **Custom fusion heads**: 3 small networks trained on top of the frozen
  CLIP features, from a simple average of the two modalities to token-level
  cross-modal attention.

## Files

| File | What it contains |
| --- | --- |
| `clip.API.ipynb` | Tutorial on the CLIP API: preprocessing, image and text embeddings, cosine similarity, zero-shot classification |
| `clip.example.ipynb` | The full project: data loading, baselines, fusion heads, results |
| `clip_utils.py` | All the logic used by the two notebooks (label aggregation, embedding extraction with caching, models, training loop, plots) |
| `extract_tokens.py` | Script that extracts the token-level CLIP features needed by the H3 head |
| `*.py` next to each notebook | The same notebook in jupytext `py:percent` format, for easy diffs |
| `Dockerfile`, `requirements.txt`, `docker_*.sh` | Container with pinned dependencies (CPU PyTorch), from `class_project/project_template` |
| `test/test_docker_template.py` | Runs both notebooks end to end inside Docker |

## Notebook Overview

### `clip.API.ipynb`

1. What CLIP is: two encoders trained with a contrastive loss.
2. Load `CLIPModel` and `CLIPProcessor`.
3. Preprocess images (224x224) and texts (up to 77 tokens).
4. Compute and L2-normalize the embeddings.
5. Plot an image-text cosine similarity matrix.
6. Zero-shot classification and its sensitivity to the prompt wording.

### `clip.example.ipynb`

1. **Load the data**: build one label per post from the 3 annotators' text
   and image votes (majority vote, then the Xu & Mao 2017 merge rule), make a
   stratified 80/10/10 split, and extract (or load) the CLIP embeddings.
2. **Evaluation protocol**: macro F1 as the main metric, hyperparameters
   chosen on validation, and the same training budget for every trained
   model (class-weighted cross-entropy, AdamW, batch 128, learning rate 1e-3,
   weight decay 1e-2, 20 epochs, best validation epoch kept).
3. **Baseline 0**: majority class.
4. **Baseline 1**: zero-shot CLIP, averaging the image and text probabilities.
5. **Baseline 2**: logistic regression on `image`, `text`, and `concat`
   embeddings.
6. **Custom fusion heads**:

   | Head | Input | Idea |
   | --- | --- | --- |
   | H1 Stack | 768-d embeddings | Standardize, average image and text, linear layer |
   | H2 Gated fusion | 768-d embeddings | Text is the anchor; the image is added through a learned gate |
   | H3 Cross-attention | Token features | Image and text tokens attend to each other |

7. **Results**: training curves, results table, and confusion matrices.

Metrics: accuracy, **macro F1** (main metric, since about 2/3 of the posts
are positive), and per-class precision and recall.

## Setup

### 1. Get the Data

Download **MVSA-Multiple** from the MCR Lab page for *Sentiment Analysis on
Multi-View Social Data* (Niu et al., MMM 2016):
http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

Unzip it so that the project directory looks like this (`data/` is
git-ignored):

```text
data/
  raw/
    MVSA/
      labelResultAll.txt
      data/
        1.jpg
        1.txt
        ...
```

The first run of `clip.example.ipynb` writes:
- `data/processed/labels.csv`: one row per usable post (~17,000) with its
  cleaned text, labels, and split.
- `data/processed/clip_embeddings.npz`: the image and text embeddings.

The H3 head also needs the token features in `data/processed/clip_tokens/`.
Extract them once, inside the container:

```bash
python extract_tokens.py text
```

```bash
python extract_tokens.py image
```

`clip.API.ipynb` reads `data/processed/labels.csv`, so run
`clip.example.ipynb` at least up to the "Load the Data" section first.

The CLIP weights (~1.7 GB) are downloaded from Hugging Face on first use and
cached in `cache/hf/` (also git-ignored).

### 2. Build and Start the Container

From this directory:

```bash
./docker_build.sh
```

```bash
./docker_jupyter.sh
```

Then open http://localhost:8888 and run the notebooks. The container mounts
the whole repository, so the data and caches stay on the host.

To run both notebooks end to end, as the tests do:

```bash
pytest test/test_docker_template.py
```

### Running Time

The container uses the CPU build of PyTorch.
- Embedding extraction: about 3 hours for ~17,000 posts on a laptop CPU.
  The result is cached, so later runs load it in seconds.
- Token features for H3: a few minutes for the texts and about 2-3 hours for
  the images. An interrupted run resumes where it stopped.
- A GPU machine with a CUDA build of PyTorch finishes both in minutes.

## Results

Test-set scores from `clip.example.ipynb` (1,703 test posts: 130 negative,
441 neutral, 1,132 positive), sorted by macro F1:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| H3 Cross-attn | 0.695 | **0.560** |
| H2 Gated | 0.639 | 0.541 |
| H1 Stack | 0.596 | 0.517 |
| LogReg (concat) | 0.581 | 0.504 |
| LogReg (text) | 0.581 | 0.502 |
| LogReg (image) | 0.566 | 0.489 |
| Zero-shot (image+text) | 0.579 | 0.401 |
| Majority class | 0.665 | 0.266 |

## Use of Claude Code Skills

The project description asks to document how the `.claude/skills/notebook.*`
skills were used. TODO (team): list the skills used and for which steps.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision*, ICML 2021: https://arxiv.org/abs/2103.00020
- Hugging Face CLIP docs:
  https://huggingface.co/docs/transformers/model_doc/clip
- Niu et al., *Sentiment Analysis on Multi-View Social Data*, MMM 2016
- Xu & Mao, *MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment
  Analysis*, CIKM 2017
