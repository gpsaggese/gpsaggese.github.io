# EconML: Analyzing the Effects of Education Programs on Student Performance

This project demonstrates how to use **EconML** for causal inference—estimating the causal impact of an educational support program (`schoolsup`) on student final grades (`G3`) and identifying which demographics benefit most (heterogeneous treatment effects / CATE).

The project is designed to be **fully reproducible via Docker** so that anyone can run the notebooks without manually installing Python packages on their machine.

## Project Layout (Folder Structure):

MSML610/  
└── Fall2025/  
&nbsp;&nbsp;&nbsp;&nbsp;└── Projects/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── UmdTask59_Fall2025_EconML_Analyzing_the_Effects_of_Education_Programs_on_Student_Performance/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── econml_utils.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── econml.API.ipynb  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── econml.API.md  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── econml.example.ipynb  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── econml.example.md  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── requirements.txt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Dockerfile  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── docker_build.sh  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── docker_jupyter.sh  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── README.md

> Note: folders like `__pycache__/` and `.ipynb_checkpoints/` are generated automatically.

## Files in This Folder

- `econml.API.ipynb` – Demonstrates EconML usage + the wrapper API in `econml_utils.py`
- `econml.API.md` – Explains API design decisions and how to use the wrapper
- `econml.example.ipynb` – End-to-end causal analysis case study (ATE + CATE + subgroup ranking + robustness)
- `econml.example.md` – Written results + interpretation (includes exported plot images)
- `econml_utils.py` – Thin wrapper around EconML estimators + helper functions for ATE/CATE analysis
- `requirements.txt` – Python dependencies
- `Dockerfile` – Container definition to install dependencies and run Jupyter
- `docker_build.sh` – Convenience script to build the Docker image
- `docker_jupyter.sh` – Convenience script to run the container and start Jupyter

## .API. vs .example. files

- **`.API.` files** focus on the **programmatic interface**:
  - how the wrapper is designed,
  - how to call EconML estimators cleanly,
  - what inputs/outputs look like.
- **`.example.` files** focus on the **applied case study**:
  - causal question framing (Y/T/X/W),
  - ATE estimation,
  - CATE estimation by demographics/SES proxies,
  - “who benefits most?” subgroup ranking,
  - robustness checks.

Recommended order to run:
1) `econml.API.ipynb`  
2) `econml.example.ipynb`

## Installation & Docker Setup

### Prerequisites
- Docker Desktop installed (Mac/Windows) or Docker Engine (Linux)
- Git (to clone the repo)

### Step 1 — Clone the repository
From a terminal:

```bash
git clone <REPO_URL>
cd <REPO_ROOT>
```

### Step 2 - Navigate to this project folder
`cd class_project/MSML610/Fall2025/Projects/UmdTask59_Fall2025_EconML_Analyzing_the_Effects_of_Education_Programs_on_Student_Performance`

### Step 3 - Build the Docker image (installs requirements)
This installs everything in `requirements.txt` inside the container.

`./docker_build.sh`

(Equivalent manual command)

`docker build -t umdtask59_econml .`

### Step 4 - Run Jupyter inside the container
This starts Jupyter and mounts the current folder into the container at `/app`, so your edits persist on your machine.

`./docker_jupyter.sh`

(Equivalent manual command)

`docker run --rm -it -p 8888:8888 -v "$(pwd)":/app umdtask59_econml`

### Step 5 - Open Jupyter and run notebooks
Open in your browser:
-   http://127.0.0.1:8888

Then run:
-   `econml.API.ipynb` (API demo)
-   `econml.example.ipynb` (full case study)

To Reproduce From Scratch: use **Kernel -> Restart & Run All**