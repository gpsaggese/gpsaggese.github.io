<!-- toc -->

- [Project files](#project-files)
- [Setup and Dependencies](#setup-and-dependencies)
  * [Building and Running the Docker Container](#building-and-running-the-docker-container)

<!-- tocstop -->

# Project files

- Author: Haochen Yang <hyang227@umd.edu> UID: 121232243
- Date: 2025-12-11

This project contains the following files

- Boilerplate files to get the project run in this project directory
  - `changelog.txt`
  - `conftest.py`
  - `invoke.yaml`
  - `pip_list.txt`
  - `poetry.lock.out`
  - `pytest.ini`
  - `repo_config.yaml`
  - `tasks.py`

- `README.md`: This file
- `NLTK.API.ipynb`: a notebook describing the native API of NLTK
- `NLTK.API.md`: a description of the native API of NLTK
- `NLTK.API.py`: a python file paired with `NLTK.API.ipynb`
- `NLTK.example.ipynb`: a notebook implementing a project using NLTK
- `NLTK.example.md`: a markdown description of the project
- `NLTK.example.py`: a python file paired with `NLTK.example.ipynb`

- `dev_scripts_tutorial_openai` boilerplate files
- `devops` mostly boilerplate files
  - `devops/docker_build/pyproject.toml`: contains the dependency of the package
    in Poetry format

# Setup and Dependencies

## Building and Running the Docker Container

- Go to the top of the repo
  ```
  > cd $GIT_ROOT
  ```
- Build the thin environment
  ```bash
  > ./helpers_root/dev_scripts_helpers/thin_client/build.py
  ```
- Go to the project dir
  ```
  > cd tutorial_openai
  ```
- Activate virtual environment:
  ```bash
  > source dev_scripts_tutorial_openai/thin_client/setenv.sh
  ```
- Build Docker Image:
  ```bash
  > i docker_build_local_image --version 1.1.0
  ```
  Note that the instruction above may fail with error message `gpg: cannot open '/dev/tty': No such device or address`. Therefore, I modified the `devops/docker_build/os_packages/install_os_publishing_tools.sh` file to avoid interaction:
  ```
  # Original code: curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
  # Modification:
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nodesource.gpg
  ```
- Run Container:
  ```bash
  > i docker_bash --skip-pull --stage local --version 1.1.0
  ```
- Launch Jupyter Notebook:
  ```bash
  > i docker_jupyter --skip-pull --stage local --version 1.1.0 -d
  ```
  Note that this instruction may also fail in the docker because there is a verification asserting that `/app` ends with a number. Therefore, use this instruction instead:
  ```
  docker run -it --rm \
    -p 8888:8888 \
    -v "$(pwd):/app" \
    --entrypoint /bin/bash \
    --name umd_task66_container \
    causify/umd_classes:local-haochen-1.1.0 \
    -c "source /venv/bin/activate && \
        pip install nltk spacy && \
        jupyter labextension enable jupytext && \
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
  ```
  The dataset downloaded is too big to commit. Therefore, I ignore the `data/` directory in the `.gitignore`. 