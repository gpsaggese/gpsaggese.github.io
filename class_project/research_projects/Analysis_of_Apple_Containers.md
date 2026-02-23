# Summary

# Tool Overview: Apple Containers

- **Apple Containers** is a macOS-native, open-source container runtime
  released by Apple at WWDC 2025
- Runs Linux OCI-compatible containers (standard Docker images) directly on
  Apple Silicon using Apple's Virtualization framework

# Project Objective

Deploy a pre-trained sentiment-analysis model as a containerized inference
server and rigorously benchmark Apple Containers against Docker Desktop on an
Apple Silicon Mac, quantifying startup latency, memory footprint, CPU usage,
and inference throughput on a real NLP dataset.

# Tasks

## Task 1: Install and Explore Apple Containers

- Install the `container` CLI tool and verify the installation
- Exercise core commands side-by-side with Docker equivalents:
  - `container build` / `docker build` -- build an image
  - `container run` / `docker run` -- launch a container
  - `container exec` / `docker exec` -- open an interactive shell
  - `container run -v` / `docker run -v` -- mount a host volume
  - `container run -p` / `docker run -p` -- expose a port
- Record any incompatibilities or behavioral differences

## Task 2: Prepare the Dataset

- Download the IMDb Large Movie Review dataset from HuggingFace Datasets
  (`datasets.load_dataset("imdb")`)
- Select a balanced sample of 2,000 reviews (1,000 positive, 1,000 negative)
  for inference benchmarking
- Save the sample as a local CSV file that will be mounted into the container
  at runtime

## Task 3: Build the Inference Service

- Load the pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model
  from HuggingFace `transformers`
- Wrap it in a `FastAPI` endpoint that accepts a JSON payload `{"text": "..."}`
  and returns a predicted label and confidence score
- Write a `requirements.txt` and a `Dockerfile` that packages the service
- Confirm that the image builds successfully and can serve requests locally

## Task 4: Run Inference with Apple Containers

- Build the image using `container build`
- Launch the server with a mounted data volume and an exposed port
- Write a Python client script that sends all 2,000 reviews to the server and
  collects predictions
- Verify prediction accuracy against the ground-truth labels (expect > 90% F1
  on this pre-trained model)

## Task 5: Run the Same Setup with Docker Desktop

- Repeat Task 4 using `docker build` and `docker run` on the same machine
- Use identical image, data, and client script to ensure fair comparison
- Note any command differences or compose-file incompatibilities encountered

## Task 6: Benchmark Both Runtimes

- Measure the following metrics for both runtimes across >= 5 independent runs:

  | Metric | Measurement method |
  |--------|-------------------|
  | Startup latency | Time from `run` command to first successful health-check |
  | Peak RAM | `psutil` / Activity Monitor during inference |
  | CPU usage | `top` averaged over the inference workload |
  | Inference throughput | Requests per second from the client script |
  | Total wall-clock time | `time` around the full inference loop |

- Compute mean and standard deviation for each metric and runtime

## Task 7: Analyze and Visualize Results

- Plot grouped bar charts comparing Apple Containers vs Docker for each metric
- Summarize compatibility findings: which Docker commands/features work
  unchanged and which require modification
- Draw conclusions about when Apple Containers is preferable over Docker
  Desktop on macOS

# Bonus Ideas (Optional)

- **Docker Compose compatibility**: write a `compose.yaml` with a model-server
  service and a pre-processing sidecar; test whether `docker compose` and any
  Apple Containers compose equivalent run it unchanged
- **Native baseline**: run inference directly (no container) on the Mac and add
  it as a third data series to quantify total container overhead
- **Larger model**: swap in `bert-base-uncased` (fine-tuned on SST-2) to
  amplify memory and latency differences between runtimes
- **Multi-run statistical test**: apply a Wilcoxon signed-rank test to confirm
  that any observed throughput difference is statistically significant

# References

- Apple Containers GitHub: https://github.com/apple/containerization
- HuggingFace `imdb` dataset: https://huggingface.co/datasets/imdb
- HuggingFace model: `distilbert-base-uncased-finetuned-sst-2-english`
- `FastAPI` documentation: https://fastapi.tiangolo.com
- `psutil` for resource monitoring: https://psutil.readthedocs.io
