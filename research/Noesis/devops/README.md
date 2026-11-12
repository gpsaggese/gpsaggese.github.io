# Devops

- Docker build, Docker Compose deployment, and container run scripts for the
  `noesis_platform` image
- Most files are symlinks to `helpers_root/devops/`; the Noesis-specific ones are
  listed below

## Structure of the Dir

| Dir             | Description                                                      |
| :-------------- | :--------------------------------------------------------------- |
| `compose/`      | Docker Compose override: `noesis_api` plus a `noesis_postgres` sidecar |
| `docker_build/` | Dockerfiles and install scripts; `pyproject.toml` and `poetry.lock` are local |
| `docker_run/`   | Container entrypoint and run scripts                             |

## Description of Files

| File                                     | Description                                                   |
| :--------------------------------------- | :------------------------------------------------------------ |
| `compose/docker-compose.noesis.yml`      | Serves `research.Noesis.noesis_platform.main:app` on port 8000 with Postgres |
| `docker_build/pyproject.toml`            | Python dependencies of the image (`poetry`)                   |
| `docker_build/poetry.lock`               | Locked dependency versions                                    |
| `docker_run/run_docker_noesis.sh`        | Start `noesis_api` and its Postgres, tail logs, stop on `Ctrl-C` |

## Description of Executables

- Run the dockerized API with a given image version (default `1.0.0`):

  ```bash
  > devops/docker_run/run_docker_noesis.sh 1.0.0
  ```
