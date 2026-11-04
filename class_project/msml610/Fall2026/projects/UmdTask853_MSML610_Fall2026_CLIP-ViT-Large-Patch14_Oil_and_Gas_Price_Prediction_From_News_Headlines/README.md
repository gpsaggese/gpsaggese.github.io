# Frozen CLIP Representations for News-Conditioned Oil and Natural Gas Forecasting

**MSML 610 — Advanced Machine Learning, Fall 2026 · Group 18 · [Project issue #853](https://github.com/gpsaggese/gpsaggese.github.io/issues/853)**  
**Author:** Shashank Ashoka (`@shashk09-coder`)  
**Status:** Initial implementation and representation-level validation (November 4, 2026); forecasting evaluation pending.

## Abstract

We investigate whether semantic representations of energy-market news can improve **next-day returns** prediction for WTI crude oil and Henry Hub natural gas beyond price-history baselines. The proposed framework uses the **frozen text encoder of CLIP ViT-L/14** to transform headlines into 768-dimensional features, which will be aligned with market observations and combined with lagged returns, momentum, and volatility in a regularized forecasting model. This initial checkpoint establishes the computational environment, verifies gradient-free CLIP inference, and probes the semantic structure of headline embeddings using event-prompt cosine similarity. On three hand-written examples, the highest-scoring prompt matches the intended event. These results validate the encoding and similarity pipeline **only**; no out-of-sample forecasting results are reported at this stage.

## 1. Approach

Given a headline $x_i$, we obtain a fixed-dimensional embedding with the pretrained CLIP text encoder $f_{\theta}$:

$$
\mathbf{h}_i = f_{\theta}(x_i) \in \mathbb{R}^{768}, \qquad \nabla_{\theta}\mathcal{L} = 0.
$$

All CLIP parameters are frozen (`requires_grad_(False)`), the model is set to evaluation mode, and embeddings are extracted under `torch.inference_mode()`. Text is tokenized with `CLIPProcessor` using padding, truncation, and a maximum sequence length of 77 tokens. To inspect semantic alignment, each headline embedding is compared with an energy-event prompt embedding $\mathbf{p}_j$ through cosine similarity:

$$
S_{ij} = \frac{\mathbf{h}_i^{\top}\mathbf{p}_j}{\lVert\mathbf{h}_i\rVert_2\,\lVert\mathbf{p}_j\rVert_2}.
$$

The comparison measures **relative representation-space affinity**, not event probabilities, causal effects, or expected market returns. Forecasting with temporally aligned news and price data is the subsequent experimental stage.

## 2. Implemented Components and Validation

| Component | Current evidence |
|:--|:--|
| Runtime | Python 3.12.15; PyTorch 2.7.1+cpu; Transformers 4.53.2; scikit-learn 1.7.1; NumPy 2.5.4; pandas 3.1.0. |
| Environment | Course Docker/Jupyter workflow established; dependency imports and version reporting executed. |
| Pretrained model | `openai/clip-vit-large-patch14` loaded successfully using `CLIPModel` and `CLIPProcessor`. |
| Feature extraction | Three illustrative headlines encoded into a **3 × 768** tensor; embeddings finite and model parameters frozen. |
| Semantic probe | Three headlines compared with four event prompts; cosine-similarity matrix has shape **3 × 4** and contains finite entries. |
| Forecasting | **Not yet implemented or evaluated.** |

<img width="1451" height="591" alt="Libraries" src="https://github.com/user-attachments/assets/d6e37bcb-df1d-413e-9805-8648a9e43cce" />

*Figure 1. Runtime and dependency smoke test. Package imports complete and the observed versions are printed; this establishes the executed environment, not a benchmark result.*

<img width="1385" height="457" alt="LOADING_CLIP" src="https://github.com/user-attachments/assets/a5b0c04f-dc73-4b9b-8241-6562040e5e12" />

*Figure 2. Loading the pretrained CLIP ViT-L/14 checkpoint and extracting three 768-dimensional headline vectors without gradient computation. The image-processor notice concerns a future default and does not indicate an error in this text-only test.*

### Semantic alignment probe

The following values are cosine similarities between **illustrative** headlines and **manually chosen** event prompts; values are rounded to three decimals.

| Headline | OPEC production cut | Weak oil demand | Cold-weather gas demand | Oil supply disruption |
|:--|--:|--:|--:|--:|
| OPEC announces a reduction in oil production. | **0.936** | 0.804 | 0.727 | 0.809 |
| Oil demand weakens as economic activity slows. | 0.804 | **0.877** | 0.761 | 0.811 |
| Cold weather increases demand for natural gas. | 0.724 | 0.741 | **0.958** | 0.749 |

<img width="1301" height="212" alt="Prompt-with-clip" src="https://github.com/user-attachments/assets/2ea047a1-4462-4022-b568-831d06f661f4" />


*Figure 3. Text–text cosine similarities for three hand-written headlines and four event descriptions. The intended event is top-ranked in each row; substantial similarity to other prompts and lexical overlap between examples limit the strength of this sanity check.*

The diagonally aligned maxima (**0.936**, **0.877**, **0.958**) show that the frozen representations recover the intended coarse semantic associations **for these constructed examples**. They do **not** establish event-classification accuracy on a labeled corpus, cross-domain robustness, or any forecasting improvement.

## 3. Reproduction

From the project directory in the course repository:

```bash
./docker_build.sh
./docker_jupyter.sh
```

In JupyterLab, run `check_dependencies.ipynb` to inspect the installed environment, followed by `clip.API.ipynb` to load CLIP, encode the three sample headlines, and reproduce the similarity table. The paired `.py` notebooks support the repository's Jupytext workflow. The model weights are retrieved from Hugging Face and cached locally in `.cache/huggingface/`.

**Reproducibility note.** `environment_versions.txt` records the successful runtime; however, the current `requirements.txt` and `Dockerfile` do **not yet pin and install** all packages required by `clip.API.ipynb` (notably PyTorch, Transformers, and scikit-learn). A fresh container build therefore should **not** be assumed to reproduce the notebook without adding these dependencies to the image configuration. The notebook documents PyTorch 2.7.1 CPU, Transformers 4.53.2, and scikit-learn 1.7.1 as tested versions.

## 4. Planned Forecasting Evaluation

The next phase will (i) collect timestamped energy headlines and WTI/Henry Hub series, (ii) align headlines to the information available before each prediction cutoff, (iii) aggregate daily CLIP text features and combine them with price-derived covariates, and (iv) fit a regularized predictor using **chronological train/validation/test splits** and training-only preprocessing. Expanding-window tuning will be used where appropriate.

Evaluation will compare **zero-return**, **historical-mean**, and **price-only** baselines with the **price + CLIP** model, including a **shuffled-headline control** to probe whether information is temporally meaningful. Planned metrics are RMSE, out-of-sample $R^2$, directional accuracy, and bootstrap confidence intervals for RMSE improvement over the price-only baseline. Particular attention will be paid to news publication timestamps, weekends and market closures, missing-news days, and temporal leakage. No forecasting performance claims are made before this evaluation is complete.

## 5. Repository Contents

- `clip.API.ipynb` / `clip.API.py` — frozen CLIP text encoding and event-prompt similarity walkthrough.
- `check_dependencies.ipynb` / `check_dependencies.py` — environment checks and package-version capture.
- `environment_versions.txt` — observed versions from the executed environment.
- `Dockerfile`, `docker_*.sh`, `requirements.txt` — course container tooling and dependencies.
- `assets/readme/` — figures documenting the current implementation checkpoint.

## References

1. Radford *et al.*, **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021. [Paper](https://arxiv.org/abs/2103.00020).
2. Hugging Face, **openai/clip-vit-large-patch14**. [Model card](https://huggingface.co/openai/clip-vit-large-patch14).
3. Hugging Face, **Transformers CLIP model documentation**. [API documentation](https://huggingface.co/docs/transformers/v4.53.2/model_doc/clip).
