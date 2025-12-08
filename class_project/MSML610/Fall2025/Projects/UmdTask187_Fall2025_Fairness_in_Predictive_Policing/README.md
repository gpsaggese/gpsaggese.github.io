# Project: Fairness in Predictive Policing - MSML610 Fall 2025

**Project Tag:** Fall2025_Fairness_in_Predictive_Policing
**GitHub Issue:** #187

## Project Objective
Develop a predictive policing model (Gradient Boosted Trees) to forecast crime hotspots while integrating **Fairlearn**  to specifically mitigate **intersectional bias** (Race $\times$ Income) in the predictions. This PR serves as the midterm structural check-in.

---

## Setup and Execution Instructions

The project is containerized using Docker to ensure a reproducible environment.

### 1. Prerequisites
You must have **Docker** installed and configured on your system.

### 2. To Build the Docker Image

Navigate to the repository root directory (`umd_classes`) and run the following command. This will use the `Dockerfile` and `requirements.txt` to install all necessary dependencies, including `fairlearn`.

```bash
docker build -t fairness-pp-umdtask187 .


# GitHub & Docker Tutorial DATA605 style

## Run instructions

1. Navigate to this path in terminal - ```class_project/instructions/tutorial_template/docker_simple```

2. Build the container
   ```bash
   > ./docker_build.sh
   ```
3. Start the Jupyter inside the container
   ```bash
   > ./docker_jupyter.sh
   ```

4. Open in browser
   - Go to [http://localhost:8888](http://localhost:8888) in your web browser

The Jupyter notebook will open in the path - ```class_project/MSML610/Fall2025/Projects```