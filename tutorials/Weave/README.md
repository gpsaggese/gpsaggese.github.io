# W&B Weave Tutorial

Learn how to trace, evaluate, and compare machine learning models using
[W&B Weave](https://weave-docs.wandb.ai/) in 60 minutes.

## Quick Start

From the repository root:

```bash
cd tutorials/Weave
./docker_build.sh
./docker_jupyter.sh
```

Set the `WANDB_API_KEY` environment variable before running the notebooks.
Obtain an API key from [Weights & Biases](https://wandb.ai/authorize).

## Notebooks

Run the notebooks in the following order:

1. [W&B Weave API Overview](weave.API.ipynb) (20 minutes):
   Learn operations, tracing, datasets, models, scorers, and evaluations.
2. [FOMC Tone and Treasury Yield Analysis](weave.example.ipynb) (25 minutes):
   Compare hawkish-dovish classifiers and analyze the relationship between
   monetary policy tone changes and two-year Treasury yield changes.

Allow 5 minutes for setup and 10 minutes for introductory reading.

## Resources
- [W&B Weave documentation](https://weave-docs.wandb.ai/)
- [Docker script documentation](docker_scripts.README.md)
- [Tutorial guidelines](../README.gp.md)