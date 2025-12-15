# How to load Data for this project


### Step 1: Go to Kaggle

Visit: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

### Step 2: Download the Data

Click the "Download" button to get `archive.zip` or `hourly-energy-consumption.zip`

### Step 3: Extract the File

Extract the ZIP file. You'll find `PJME_hourly.csv` inside.

### Step 4: Place It Here

Copy `PJME_hourly.csv` to this `data/` folder:

```
electricity_forecasting_final/
└── data/
    └── PJME_hourly.csv  ← PUT THE FILE HERE!
```

### Step 5: Verify

Check the file is in place:

```bash
ls -lh data/PJME_hourly.csv
```

You should see a file around 3-4 MB.

---

## Once the dataset is in place, run:

```bash
./docker/build.sh
./docker/run_jupyter.sh
```

Then open http://localhost:8888 and run the notebooks.
