- **Objective**: Run the below tutorials and submit a report detailing your
  understanding. Your report should not exceed 3 pages.

# Running with invoke

- Run
  ```
  > cd  ~/src/umd_classes1/msml610/tutorials                                                                                                                                       > i docker_jupyter --skip-pull --stage local --version 1.0.0
  > i docker_jupyter --skip-pull --stage local --version 1.0.0 -p 5012
  ```

- If everything works correctly you should see
  ```
  ...
  [I 2026-01-23 23:01:05.740 ServerApp] http://localhost:5012/lab
  [I 2026-01-23 23:01:05.740 ServerApp]     http://127.0.0.1:5012/lab
  [I 2026-01-23 23:01:05.740 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
  ```

- Go with your browser to http://localhost:5012/lab and you should see the
  Jupyter lab

# Running directly with docker

- Ensure your thin env is activated
   ```bash
   > source setenv.sh
   ```

- Navigate to:
  ```
  > cd umd_classes/msml610
  ```

- ## `Add the following 2 lines JUST BELOW THE`

   ## ***`echo "##> $FILE_NAME"`*** `code in the run_jupyter_server.sh`

  ```
  source /venv/bin/activate
  cd /workspace
  ```

4. Add the following code JUST below the poetry install --no-root line of code in the install_python_packages.sh file

  ```py
  > pip install jupyter jupyterlab jupytext ipykernel
  ```

5. ## Build the Docker image

  ```
  docker build --no-cache -f tutorials/devops/docker_build/dev.Dockerfile --build-arg POETRY_MODE=update -t msml610assignmentimage .
  ```

### NOTE: Don’t forget to copy the period at the end of the code.

6. ## Run

```
cd ..
```

7. ## Run Docker Jupyter

```
docker run --rm \
  --entrypoint "" \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -e PORT=8888 \
  msml610assignmentimage \
  /workspace/msml610/tutorials/devops/docker_run/run_jupyter_server.sh

```

8. Before you start executing the Jupyter notebook code cells, add the following lines of code at the very top of each notebook you execute

```
import sys, os
sys.path.append("/workspace")
sys.path.append("/workspace/helpers_root") sys.path.append("/workspace/msml610/tutorials")
os.environ["CSFY_GIT_ROOT_PATH"] = "/workspace/msml610"

```

After that, replace the below line in [msml610\_utils](https://github.com/gpsaggese-org/umd_classes/blob/master/msml610/tutorials/msml610_utils.py).py

```
import helpers.hdbg as hdbg
```

with

```
import helpers_root.helpers.hdbg as hdbg
```

9. Execute the Jupyter notebooks

**Troubleshooting**:

If any of the imported packages throw a ModuleNotFoundError, run the below
command in a jupyter code cell

```
!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet PACKAGENAME)"
```
