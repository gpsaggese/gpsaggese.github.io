# AutoGen Tutorial

- This folder contains the setup for running AutoGen tutorials within a
  containerized environment

## Quick Start
- From the root of the repository, change your directory to the Autogen tutorial
  folder:
  ```bash
  > cd tutorials/Autogen
  ```

- Once the location has been changed to the repo run the command to build the
  image to run dockers:
  ```bash
  > ./docker_build.sh
  ```

- Once the docker has been built you can then go ahead and run the container and
  launch jupyter notebook using the created image using the command:
  ```bash
  > ./docker_jupyter.sh
  ```

- Once the `./docker_jupyter.sh` script is running, follow this sequence to
  explore the tutorials:
  1. **`autogen.API.ipynb`**: Start here to master the fundamental commands and
     basic configurations of the AutoGen framework.
  2. **`Autogen.example.ipynb`**: Proceed to this notebook to explore more
     complex, multi-agent scenarios and advanced problem-solving techniques.

- For more informations on the Docker build system refer to [Project template
  readme](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md)
