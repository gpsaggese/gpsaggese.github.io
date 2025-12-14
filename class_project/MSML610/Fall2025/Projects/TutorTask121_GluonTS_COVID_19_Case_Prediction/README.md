# COVID-19 Case Prediction using GluonTS

**Time Series Forecasting**

Probabilistic forecasting of COVID-19 daily cases using three GluonTS models: DeepAR, SimpleFeedForward, and DeepNPTS.

---

## Getting Started

### Prerequisites
- Docker installed and running
- 8GB RAM minimum

### Build and Run

#### Build the Docker Image

```bash
./docker_build.sh
```

**Expected Output:**
```
Building Docker Image: gluonts-covid
==========================================
[+] Building 45.2s (12/12) FINISHED
 => [internal] load build definition
 => => transferring dockerfile
 => [internal] load .dockerignore
 => [1/6] FROM docker.io/library/python:3.10-slim
 => [2/6] RUN apt-get update && apt-get install -y build-essential
 => [3/6] WORKDIR /workspace
 => [4/6] COPY requirements.txt .
 => [5/6] RUN pip install --no-cache-dir -r requirements.txt
 => [6/6] COPY . /workspace
 => exporting to image
 => => exporting layers
 => => writing image sha256:abc123...
 => => naming to docker.io/library/gluonts-covid

Docker image built successfully

Next steps:
  Run Jupyter: ./docker_jupyter.sh
  Run bash: ./docker_bash.sh
```

**Build time:** 1-2 minutes for the first time, <30 seconds for subsequent builds

---

#### Run Jupyter Notebook

```bash
./docker_jupyter.sh
```

**Expected Output:**
```
Starting Jupyter Notebook Server
==========================================
URL: http://localhost:8888
Press Ctrl+C to stop
==========================================

[I 2025-12-14 12:00:00.123 ServerApp] Jupyter Server 2.x.x is running at:
[I 2025-12-14 12:00:00.123 ServerApp] http://localhost:8888/tree
[I 2025-12-14 12:00:00.123 ServerApp] Use Control-C to stop this server
```

**Open your browser** to `http://localhost:8888` and navigate to:
- `GluonTS.API.ipynb` - Learn model APIs
- `GluonTS.example.ipynb` - Complete application

---

#### Access Interactive Shell (Optional)

```bash
./docker_bash.sh
```

**Expected Output:**
```
Starting Interactive Bash Shell
==========================================
Working directory: /workspace
Type 'exit' to leave
==========================================

root@abc123:/workspace#
```

Useful for running Python scripts directly or debugging.

---

### File Organization

```
TutorTask121_GluonTS_COVID_19_Case_Prediction/
│
├── Main Notebooks (Start Here)
│   ├── GluonTS.API.ipynb           # Model API demonstrations
│   ├── GluonTS.API.md              # API documentation
│   ├── GluonTS.example.ipynb       # Complete application
│   └── GluonTS.example.md          # Application guide
│
├── Utility Modules
│   ├── GluonTS_utils_data_io.py            # Load COVID data
│   ├── GluonTS_utils_preprocessing.py      # Clean and prepare data
│   ├── GluonTS_utils_gluonts.py            # GluonTS formatting
│   ├── GluonTS_utils_evaluation.py         # Metrics and plots
│   ├── GluonTS_utils_notebook_loader.py    # Quick data loader
│   └── GluonTS_utils_models.py             # Model wrappers
│
├── Data Files
│   └── data/
│       ├── cases.csv               # JHU COVID-19 cases
│       ├── deaths.csv              # JHU COVID-19 deaths
│       ├── mobility.csv            # Google Mobility data
│       └── vaccine.csv             # CDC vaccines (not used)
│
├── Docker Setup
│   ├── Dockerfile                  # Container configuration
│   ├── docker_build.sh             # Build script
│   ├── docker_jupyter.sh           # Jupyter launcher
│   └── docker_bash.sh              # Shell access
│
└── requirements.txt                # Python dependencies
```

---

### Model Comparison

| Model | External Features | Training Time | Best Use Case |
|-------|------------------|---------------|---------------|
| **DeepAR** | Yes (deaths, mobility, CFR) | 3-4 min | Complex patterns, highest accuracy |
| **SimpleFeedForward** | No | 30-60 sec | Quick baselines, stable trends |
| **DeepNPTS** | Yes (deaths, mobility, CFR) | 3-4 min | Regime changes, distribution shifts |

---

## Data Pipeline

```mermaid
flowchart TB
    A[Raw Data Sources] --> B[JHU Cases & Deaths]
    A --> C[Google Mobility]
    A --> D[CDC Vaccines]
    
    B --> E[Load & Parse]
    C --> E
    D --> F[Keep for Future]
    
    E --> G[Aggregate to National Level]
    G --> H[Calculate Moving Averages]
    H --> I[Engineer Features]
    
    I --> J[Daily_Cases_MA7]
    I --> K[Daily_Deaths_MA7]
    I --> L[Cumulative_Deaths]
    I --> M[CFR]
    I --> N[6 Mobility Metrics]
    
    J --> O[Train/Test Split]
    K --> O
    L --> O
    M --> O
    N --> O
    
    O --> P[GluonTS Format]
    P --> Q[Ready for Training]
```

### Features Used

1. **Target:** Daily COVID-19 cases (7-day moving average)
2. **Deaths Features:** Daily deaths (MA7), cumulative deaths, CFR
3. **Mobility Features:** Retail, grocery, parks, transit, workplaces, residential

---


**Metrics Explained:**
- **MAE:** Average absolute difference (lower = better)
- **RMSE:** Penalizes large errors more (lower = better)
- **MAPE:** Percentage error, scale-independent (lower = better)
- **CRPS:** Probabilistic forecast quality (lower = better)

---

## Troubleshooting

### Docker Issues

**Problem:** Port 8888 already in use

**Solution:** Stop existing Jupyter or change port:
```bash
# Edit docker_jupyter.sh, line 14:
-p 8889:8888  # Use 8889 instead
```

---

**Problem:** Docker build fails

**Solution:** Ensure Docker is running:
```bash
docker info
```

---

### Training Issues

**Problem:** MPS (Apple GPU) not supported error

**Solution:** Should already be handled! The shell scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1` to use CPU for unsupported operations.

**Expected message:**
```
GPU available: True (mps), used: True
WARNING: Using CPU fallback for unsupported MPS operations
```

This is normal and won't significantly impact performance.

---

**Problem:** Out of memory

**Solution:** Reduce batch size in notebook:
```python
batch_size = 16  # Instead of 32
```

---

### Data Issues

**Problem:** Data files not found

**Solution:** Verify data directory:
```bash
ls data/
# Should show: cases.csv, deaths.csv, mobility.csv, vaccine.csv
```

---

## Additional Resources

### GluonTS Documentation
- Official docs: https://ts.gluon.ai/
- GitHub: https://github.com/awslabs/gluonts

### Research Papers
- DeepAR: https://arxiv.org/abs/1704.04110
- DeepNPTS: https://arxiv.org/abs/1906.05264

### COVID-19 Data Sources
- JHU CSSE: https://github.com/CSSEGISandData/COVID-19
- Google Mobility: https://www.google.com/covid19/mobility/
- CDC Data: https://covid.cdc.gov/covid-data-tracker/

---

## Technical Specifications

**Python:** 3.10+  
**Framework:** GluonTS 0.14.0 with PyTorch backend  
**Hardware:** CPU-optimized, Apple Silicon compatible  
**Docker:** Multi-stage build with slim base image  
**Data:** 1,143 days training, 14 days testing

---

## Quick Command Reference

```bash
# Build
./docker_build.sh

# Run Jupyter (default)
./docker_jupyter.sh

# Interactive shell
./docker_bash.sh

# Stop container
# Press Ctrl+C in terminal where Jupyter is running

# View running containers
docker ps

# Remove all containers
docker rm $(docker ps -aq)
```

---

