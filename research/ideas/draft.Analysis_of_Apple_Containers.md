# Analysis of Apple Containers

## Status

- **Status:**: draft
- **Complete Specs:**: 60%
- **Assignee:**: TBD

## Core Idea

- **Apple Containers** is a macOS-native, open-source container runtime released by
  Apple at WWDC 2025
  - Runs Linux OCI-compatible containers (standard Docker images) directly on Apple
    Silicon
  - Uses Apple's Virtualization framework
- Project objective: deploy a pre-trained sentiment-analysis model as a
  containerized inference server
  - Benchmark Apple Containers against Docker Desktop on an Apple Silicon Mac
  - Quantify:
    - Startup latency
    - Memory footprint
    - CPU usage
    - Inference throughput
  - Use a real NLP dataset

## Formalization

### Benchmark Metrics

| Metric                | Measurement method                                       |
| --------------------- | -------------------------------------------------------- |
| Startup latency       | Time from `run` command to first successful health-check |
| Peak RAM              | `psutil` / Activity Monitor during inference             |
| CPU usage             | `top` averaged over the inference workload               |
| Inference throughput  | Requests per second from the client script               |
| Total wall-clock time | `time` around the full inference loop                    |

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. _When is Apple Containers preferable over Docker Desktop on macOS?_
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Docker Compose compatibility**: write a `compose.yaml` with a model-server
  service and a pre-processing sidecar
  - Test whether `docker compose` and any Apple Containers compose equivalent run
    it unchanged
- **Native baseline**: run inference directly (no container) on the Mac
  - Add it as a third data series to quantify total container overhead
- **Larger model**: swap in `bert-base-uncased` (fine-tuned on SST-2)
  - Amplify memory and latency differences between runtimes
- **Multi-run statistical test**: apply a Wilcoxon signed-rank test
  - Confirm that any observed throughput difference is statistically significant

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: install and explore Apple Containers
  - Install the `container` CLI tool and verify the installation
  - Exercise core commands side-by-side with Docker equivalents:
    - `container build` / `docker build`: build an image
    - `container run` / `docker run`: launch a container
    - `container exec` / `docker exec`: open an interactive shell
    - `container run -v` / `docker run -v`: mount a host volume
    - `container run -p` / `docker run -p`: expose a port
  - Record any incompatibilities or behavioral differences

- Milestone 2: prepare the dataset
  - Download the IMDb Large Movie Review dataset from HuggingFace Datasets
    (`datasets.load_dataset("imdb")`)
  - Select a balanced sample of 2,000 reviews (1,000 positive, 1,000 negative) for
    inference benchmarking
  - Save the sample as a local CSV file that will be mounted into the container at
    runtime

- Milestone 3: build the inference service
  - Load the pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model
    from HuggingFace `transformers`
  - Wrap it in a `FastAPI` endpoint
    - Accepts a JSON payload `{"text": "..."}`
    - Returns a predicted label and confidence score
  - Write a `requirements.txt` and a `Dockerfile` that packages the service
  - Confirm that the image builds successfully and can serve requests locally

- Milestone 4: run inference with Apple Containers
  - Build the image using `container build`
  - Launch the server with a mounted data volume and an exposed port
  - Write a Python client script that sends all 2,000 reviews to the server and
    collects predictions
  - Verify prediction accuracy against the ground-truth labels
    - Expect > 90% F1 on this pre-trained model

- Milestone 5: run the same setup with Docker Desktop
  - Repeat Milestone 4 using `docker build` and `docker run` on the same machine
  - Use identical image, data, and client script to ensure fair comparison
  - Note any command differences or compose-file incompatibilities encountered

- Milestone 6: benchmark both runtimes
  - Measure the metrics in `### Benchmark Metrics` (Formalization) for both
    runtimes across >= 5 independent runs
  - Compute mean and standard deviation for each metric and runtime

- Milestone 7: analyze and visualize results
  - Plot grouped bar charts comparing Apple Containers vs Docker for each metric
  - Summarize compatibility findings
    - Which Docker commands/features work unchanged
    - Which Docker commands/features require modification
  - Draw conclusions about when Apple Containers is preferable over Docker Desktop
    on macOS

## References

- Apple Containers GitHub: https://github.com/apple/containerization
- HuggingFace `imdb` dataset: https://huggingface.co/datasets/imdb
- HuggingFace model: `distilbert-base-uncased-finetuned-sst-2-english`
- `FastAPI` documentation: https://fastapi.tiangolo.com
- `psutil` for resource monitoring: https://psutil.readthedocs.io
