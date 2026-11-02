# Tracing and Evaluating Hawkish-Dovish Models for FOMC Text (MSML610, Fall 2026)

This project uses [W&B Weave](https://weave-docs.wandb.ai/) to trace, evaluate,
and compare models that label Federal Reserve (FOMC) text as **hawkish**,
**dovish**, or **neutral**. It then tests whether the tone of FOMC statements
relates to the change in the 2-year Treasury yield on the announcement day.

- **Authors**: Mitali Purwar (@MitaliPurwar21), Rachel Bubeck (@rbubeck)
- **Issue**: [#657](https://github.com/gpsaggese/gpsaggese.github.io/issues/657)
- **Project description**:
  [WB_Weave_Project_Description.md](../../../../project_descriptions/MSML610/WB_Weave_Project_Description.md)

## 1. Overview

| Task | What it does | Where |
|---|---|---|
| Ingest the data | Load the labeled FOMC sentences, download the 2015–2024 FOMC statements and the daily 2-year Treasury yields, and cache everything on disk | `weave.example.ipynb` |
| Define the problem | Classify sentences with macro-F1 on `test.csv`; fit a TF-IDF + logistic regression baseline on `train.csv`; define statement tone as the share of hawkish sentences minus the share of dovish sentences | `weave.example.ipynb` |
| Trace the models | Wrap the baseline, FOMC-RoBERTa, and a zero-shot classifier as `weave.Model` classes with a traced `predict` | `weave_utils.py`, `weave.example.ipynb` |
| Evaluate with Weave | Publish the test sentences as a `weave.Dataset` and run `weave.Evaluation` with an accuracy scorer and a custom macro-F1 `weave.Scorer` | `weave_utils.py`, `weave.example.ipynb` |
| Link tone to yields | Pick the model by macro-F1, then measure the Spearman correlation (with a bootstrap 95% CI) between the change in tone and the yield change | in progress |
| Visualize and report | Compare the evaluations in the Weave UI and plot tone change against yield change | in progress |

## 2. Project Structure

```
UmdTask657_MSML610_Fall2026_WB_Weave_Project1/
├── README.md                  # This file
├── progress.md                # Experiment log: status, findings, and history
├── weave.API.ipynb / .py      # Tutorial on the Weave API with a toy classifier
├── weave.example.ipynb / .py  # The FOMC project, end to end
├── weave_utils.py             # Tone models, scorers, and evaluation helpers
├── requirements.txt           # Python packages installed in the container
├── Dockerfile                 # Container image (class template, Python 3.12 slim)
├── docker_*.sh, utils.sh      # Class template scripts to build and run the container
├── docker_build.version.log   # Package versions recorded at the last build
├── .env                       # API keys (git-ignored, see below)
└── data/                      # Downloaded and cached data (git-ignored)
    ├── raw/                   # Labeled sentences, FOMC statements, Treasury yields
    ├── processed/             # Cleaned train/test splits and aligned tables
    └── hf_cache/              # Hugging Face model weights
```

Each notebook is paired with a `.py` file through
[jupytext](https://jupytext.readthedocs.io/), so changes can be reviewed and
merged as plain Python.

## 3. Prerequisites

1. **Docker.** Everything runs inside the project container.
   - On macOS with [Colima](https://github.com/abiosoft/colima), the scripts
     need two extra steps:
     - The scripts default to Apple's `container` tool on macOS. Switch them to
       Docker with `echo 'export DOCKER_ENGINE=docker' >> ~/.zshrc`.
     - Install the buildx plugin: `brew install docker-buildx`, then link it
       with `mkdir -p ~/.docker/cli-plugins && ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" ~/.docker/cli-plugins/docker-buildx`.
   - On Windows, use Docker Desktop with WSL integration enabled for the
     Linux distribution, and keep the repo inside the WSL file system (for
     example `~/src`).
2. **A W&B account** for Weave, with an API key from
   <https://wandb.ai/authorize>.
3. **A Hugging Face account with access to
   [`gtfintechlab/FOMC-RoBERTa`](https://huggingface.co/gtfintechlab/FOMC-RoBERTa).**
   The model is gated: request access on the model page (approval is manual
   and can take time), then create a **read** token at
   <https://huggingface.co/settings/tokens>. A fine-grained token also needs
   the permission to read public gated repos you have access to.

## 4. Setup

From the project folder:

```bash
cd class_project/msml610/Fall2026/projects/UmdTask657_MSML610_Fall2026_WB_Weave_Project1
```

Create a `.env` file with the two keys. It is listed in `.gitignore`, so it is
never committed:

```bash
WANDB_API_KEY=<your W&B API key>
HF_TOKEN=<your Hugging Face read token>
```

Build the image (the first build downloads PyTorch and takes several minutes):

```bash
./docker_build.sh
```

## 5. Running the Notebooks

Start Jupyter inside the container:

```bash
./docker_jupyter.sh
```

Then open <http://localhost:8888>. The project folder opens by default.

Run each notebook with **Kernel → Restart Kernel and Run All Cells**.

### `weave.API.ipynb`

A self-contained tour of `weave.init`, `@weave.op`, `weave.Dataset`,
`weave.Model`, and `weave.Evaluation`. It uses a keyword classifier on 12
hand-written sentences, so it runs in a few seconds and needs no downloads.
Results go to the Weave project `msml610-weave-api`.

### `weave.example.ipynb`

The full project. Results go to the Weave project `msml610-weave-fomc`.

- **First run**: downloads the labeled sentences, about 85 FOMC statements,
  11 years of Treasury yield files, and two Hugging Face models (BART-MNLI,
  about 1.6 GB, and FOMC-RoBERTa, about 1.4 GB). Expect 15–20 minutes,
  mostly for the downloads and the zero-shot evaluation.
- **Later runs**: read everything from `data/`. Delete a file in `data/` to
  download it again.
- The zero-shot model takes about 1 second per sentence on CPU, so its
  evaluation on the 496 test sentences takes about 8 minutes.

## 6. Models and Evaluation

| Model | Description | Where it comes from |
|---|---|---|
| `TfidfLogRegModel` | TF-IDF (unigrams and bigrams) + logistic regression | Fit on `train.csv` |
| `FomcRobertaModel` | RoBERTa-large fine-tuned on FOMC sentences | [`gtfintechlab/FOMC-RoBERTa`](https://huggingface.co/gtfintechlab/FOMC-RoBERTa) |
| `ZeroShotModel` | Zero-shot NLI classifier with a short description of each label | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) |

Every model's `predict(sentence)` returns
`{"label": 0/1/2, "label_name": ..., "confidence": ...}`, using the dataset's
encoding: 0 = dovish, 1 = hawkish, 2 = neutral.

The evaluation runs every model on the 496 test sentences with two scorers:

- `accuracy_scorer`: an `@weave.op` that marks each prediction right or wrong
- `MacroF1Scorer`: a `weave.Scorer` whose `summarize()` computes macro-F1 and
  per-class F1 over all rows. A class the model never predicts counts with an
  F1 of 0.

Macro-F1 is the selection metric, as the project description requires. The
models are never tuned on the test set.

## 7. Docker Customizations

The container follows the class template with these changes:

- **Base image**: the template default, Python 3.12 slim. The Dockerfile is
  unchanged.
- **Image name**: `docker_name.sh` sets `IMAGE_NAME=umd_msml610_wb_weave_project1`,
  so the image does not clash with other projects built from the template.
- **`requirements.txt`** adds `weave`, `scikit-learn`, `python-dotenv`,
  `torch`, and `transformers` to the template packages. It also adds the
  PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`),
  so x86 machines install the CPU build of PyTorch instead of the much larger
  CUDA build. The project runs on CPU only.
- **`.gitignore` / `.dockerignore`**: exclude `.env` (API keys) and `data/`
  (downloaded data and model weights).

## 8. Data Sources

| Data | Source | Cached in |
|---|---|---|
| Labeled FOMC sentences (train/test) | [Trillion Dollar Words](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | `data/raw/`, `data/processed/` |
| FOMC statements 2015–2024 | [Federal Reserve FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | `data/raw/fomc_statements/` |
| Daily 2-year Treasury par yields | [U.S. Treasury](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve) | `data/raw/treasury_yields/` |

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: You are trying to access a gated repo` | Access to FOMC-RoBERTa is not approved yet, or `HF_TOKEN` is missing from `.env` |
| `WANDB_API_KEY is missing` | Add the key to `.env` in the project folder |
| `BuildKit is enabled but the buildx component is missing` | Install the buildx plugin (see Prerequisites) |
| `docker_jupyter.sh` prints `denied: requested access to the resource is denied` | Harmless: the script tries to pull the image from Docker Hub first, then uses the local build |
| Port 8888 is busy | `./docker_jupyter.sh -p 8890`, then open <http://localhost:8890> |
