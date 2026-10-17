# GitHub & Docker Tutorial for MSML610

## Run instructions

1. Navigate to the tutorial directory:

   ```bash
   cd class_project/instructions/tutorial_template/tutorial_github_data605_style
   ```

2. Build the Docker container:

   ```bash
   ./docker_build.sh
   ```

3. Start Jupyter inside the container:

   ```bash
   ./docker_jupyter.sh
   ```

4. Open [http://localhost:8888](http://localhost:8888) in your browser.

The notebook environment uses the directory mounted by the Docker
Jupyter script. Use the `-d` option to specify a different host directory
if needed.
