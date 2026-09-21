## Test Organization

Tests for msml610 tutorials are organized in a two-tier structure:

- **Docker Container Tests**: Each tutorial directory has a `test/` subdirectory
  with `test_docker_all.py` (e.g.,
  `msml610/tutorials/L03_knowledge_representation/test/test_docker_all.py`).
  These tests inherit from `hdoctest.DockerTestCase` and verify that:
  - `docker_build.sh` builds the container successfully
  - `docker_cmd.sh` can execute commands inside the container
  - Individual notebooks execute without errors inside the container
  - Optionally, static HTML renders of notebooks (with per-cell anchors) are
    generated

- **Test Types via DockerTestCase**:
  - **Container build tests** (`test_docker_build`): Verify the Docker image
    builds successfully. Runs first (marked with `@pytest.mark.order(1)`).
  - **Container execution tests** (`test_docker_cmd`): Verify arbitrary shell
    commands can run inside the container (marked with `@pytest.mark.order(2)`).
  - **Interactive bash tests** (`test_docker_bash`): Optional interactive bash
    access tests via `docker_bash.sh` (marked with `@pytest.mark.order(3)`).
  - **Notebook execution tests** (`test1`, `test2`, etc.): Each test runs a
    specific notebook via `run_nbconvert.py` (through
    `run_notebook(..., use_docker_cmd=False)`), which executes the notebook in
    the container and generates the HTML. Tests are numbered to
    match notebook numbering (e.g., `test3` runs `L03_03_*.ipynb`).
  - **HTML generation tests** (`test1_html`, `test2_html`, etc.): Optional tests
    to regenerate static HTML renderings of notebooks with working per-cell
    anchors and ipywidgets state. Marked with `@pytest.mark.slow` and only run
    explicitly by name.

- Find the tests that build a Docker container

```
> pytest msml610/tutorials/ --collect-only -qqq | grep test_docker_build
msml610/tutorials/L03_knowledge_representation/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L05_statistical_learning/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L06_bayesian_networks/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L07_prob_programming/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L08_causal_inference/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L09_kalman_filter/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L09_multi_armed_bandits/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L10_causal_discovery/test/test_docker_all.py::Test_docker::test_docker_build
msml610/tutorials/L12_reinforcement_learning/test/test_docker_all.py::Test_docker::test_docker_build
```

## docker_build_all.sh

Batch build utility for all tutorial Docker images.

- **Purpose**: Iterates through all tutorial directories under
  `msml610/tutorials/` and builds each one's Docker image by executing its
  `docker_build.sh` script.
- **Resilient failure handling**: If one tutorial's build fails, the script
  continues building the rest instead of stopping. A pass/fail summary is
  printed at the end showing which tutorials succeeded and which failed.
- **Argument forwarding**: Any command-line arguments passed to this script
  (e.g., `--no-cache`, `-v`) are forwarded to each individual `docker_build.sh`
  call.

**Usage**:
```bash
msml610/tutorials/docker_build_all.sh
msml610/tutorials/docker_build_all.sh --no-cache
```

## check_containers.sh

Diagnostic utility for verifying Docker image and container status.

- **Purpose**: Inspects each tutorial directory and reports on container status
  for both Apple `container` (native container runtime) and Docker.
- **Output format**: Prints a table with columns:
  - `TUTORIAL`: Tutorial directory name
  - `IMAGE`: Full Docker image name (sourced from each tutorial's
    `docker_name.sh`)
  - `APPLE_BUILT`: Whether image exists in Apple container storage
  - `APPLE_RUN`: Whether a container from that image is running (Apple)
  - `DOCKER_BUILT`: Whether image exists in Docker image storage
  - `DOCKER_RUN`: Whether a container from that image is running (Docker)
- **Isolated variable sourcing**: Uses subshell when sourcing `docker_name.sh`
  to avoid variable leakage across iterations.
- **Dual-runtime support**: Detects images/containers in both Apple's native
  container tooling and Docker, allowing inspection of tutorial containers
  across different container runtimes.

**Usage**:
```bash
msml610/tutorials/check_containers.sh
```

**Example output**:
```
TUTORIAL                       IMAGE                                        APPLE_BUILT  APPLE_RUN    DOCKER_BUILT DOCKER_RUN
L03_knowledge_representation   umd_classes2/msml610:L03_knowledge_representation  yes          no           yes          no
L06_bayesian_networks          umd_classes2/msml610:L06_bayesian_networks   yes          no           yes          yes
```
