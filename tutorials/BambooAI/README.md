# BambooAI Tutorial

This folder contains the setup for running BambooAI tutorials within a
containerized environment.

## Quick Start

From the root of the repository, change your directory to the BambooAI tutorial folder:

```bash
> cd tutorials/BambooAI
```

Once the location has been changed to the repo run the command to build the image
to run dockers:

```bash
> ./docker_build.sh
```

Once the docker has been built you can then go ahead and run the container and
launch jupyter notebook using the created image using the command:

```bash
> ./docker_jupyter.sh
```

Once the `./docker_jupyter.sh` script is running, work through the following
notebooks in order.

For more information on the Docker build system refer to [Project template
README](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks

Work through the following notebooks in order:

- [`bambooai.API.ipynb`](bambooai.API.ipynb): Core BambooAI fundamentals
  - Understanding the BambooAI framework architecture
  - Working with BambooAI classes and methods
  - Building basic agent configurations
  - Integration with language models

- [`bambooai.example.ipynb`](bambooai.example.ipynb): Real-world application
  workflow
  - End-to-end agentic application example
  - Practical problem-solving with BambooAI
  - Advanced agent interactions and workflows
  - Best practices and patterns

- [`bambooai_utils.py`](bambooai_utils.py): Utility functions supporting the
  tutorial notebooks

## Changelog

- 2026-03-15: Initial release
