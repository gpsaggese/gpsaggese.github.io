# CLIP Tutorial

- Learn CLIP (`openai/clip-vit-large-patch14`) in 60 minutes, applied to
  sentiment analysis of tweets with an image (MVSA-Multiple)

## Quick Start
- Download MVSA-Multiple from the
  [MCR Lab page](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
  and unzip it into `data/raw/MVSA/`
- Build the image and launch Jupyter:
  ```bash
  > cd class_project/msml610/Fall2026/projects/UmdTask593_MSML610_Fall2026_CLIP_ViT_Large_Patch14
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```
- For more information on the Docker build system refer to
  [Project template readme](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks
1. [`clip.API.ipynb`](clip.API.ipynb): what CLIP is, preprocessing,
   embeddings, cosine similarity, and zero-shot classification
2. [`clip.example.ipynb`](clip.example.ipynb): sentiment of tweets with
   baselines (zero-shot, logistic regression) and 3 fusion heads on frozen
   CLIP features (best: token cross-attention, test macro F1 0.560)

- [`clip_utils.py`](clip_utils.py): all the logic used by the notebooks
- [`extract_tokens.py`](extract_tokens.py): extracts the token features for the
  H3 head (`python extract_tokens.py text`, then `image`; ~3 h on CPU)

## Use of Claude Code Skills
- TODO (team): list the `.claude/skills/notebook.*` skills used, and for what

## Changelog
- 2026-11-26: Initial release
