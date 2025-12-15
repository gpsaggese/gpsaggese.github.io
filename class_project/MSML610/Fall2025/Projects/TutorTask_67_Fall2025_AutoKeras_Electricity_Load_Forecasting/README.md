# Electricity Load Forecasting with AutoKeras

Complete project for forecasting electricity demand using AutoKeras StructuredDataRegressor.

## Quick Start

### 1. Download Dataset

**IMPORTANT:** Before running, download the dataset:

1. Go to: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
2. Download `PJME_hourly.csv` (or the archive and extract it)
3. Place the file in the `data/` folder

```
electricity_forecasting_final/
└── data/
    └── PJME_hourly.csv  ← PUT FILE HERE
```

### 2. Build Docker Container

```bash
cd electricity_forecasting_final
chmod +x docker/*.sh
./docker/build.sh
```

### 3. Start Jupyter

```bash
./docker/run_jupyter.sh
```

### 4. Open Browser

Go to: **http://localhost:8888**

### 5. Run Notebooks

---

## What This Does

1. **Loads** 20 years of electricity data (130K+ hours)
2. **Engineers** 40+ time-based features
3. **Trains** AutoKeras to find best model (10 trials)
4. **Compares** with baseline models
5. **Evaluates** using MAE, RMSE, MAPE
6. **Visualizes** predictions and errors

---

## Expected Results

- **AutoKeras MAPE**: 4-6% (excellent!)
- **Training Time**: 20-30 minutes
- **Outperforms baselines by**: 30-50%

---

## Deliverables Included

`AutoKeras.API.md` - API documentation
`AutoKeras.API.ipynb` - API tutorial  
`AutoKeras.example.md` - Project guide  
`AutoKeras.example.ipynb` - Complete project  
`autokeras_utils.py` - Utility module  

---

## Checklist

Before running:
- ☐ Dataset downloaded to `data/PJME_hourly.csv`
- ☐ Docker Desktop running
- ☐ Container built successfully

After running:
- ☐ Notebooks execute without errors
- ☐ MAPE < 10% achieved
- ☐ Visualizations generated

---
