# AutoGluon Tutorial: Predicting NYC Airbnb Prices

Learn AutoGluon's machine learning workflow by training models to predict Airbnb listing prices. The tutorial covers training, evaluation, model comparison, and storing procedures. The project includes an API introduction and a hands-on example using real Airbnb listings.

## Learning objectives

In this tutorial, you will learn how to:

- Define a tabular regression problem
- Train an AutoGluon predictor
- Generate predictions and evaluate MAE, RMSE, and R²
- Compare models using validation and test results
- Interpret feature importance and its limitations
- Save and reload a trained predictor
- Compare AutoGluon with a Linear Regression baseline.
- Investigate how feature selection affects predictions

## Project files

| File | Description |
|---|---|
| `autogluon.API.ipynb` | Introduces the main AutoGluon methods using synthetic data. |
| `autogluon.example.ipynb` | Applies AutoGluon to Airbnb prices and compares it with Linear Regression. |
| `Dockerfile` | Defines the Python environment and starts JupyterLab. |
| `requirements.txt` | Lists the Python dependencies. |
| `.dockerignore` | Excludes unnecessary files from the Docker build context. |
| `.gitignore` | Excludes local data, generated models, and temporary files from Git. |

The `data/` and `AutogluonModels/` folders are created locally and are not included in the repository.

## Requirements

- Docker Desktop
- Sufficient free disk space for the Docker image and generated models (AutoGluon recommends at least 10 GB of available disk space)

The commands below use Windows PowerShell.

## Dataset setup

Download `AB_NYC_2019.csv` from:

```powershell
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
```

From the project folder, create a data directory:

```powershell
mkdir data
```

Place the downloaded CSV at:

```text
data/AB_NYC_2019.csv
```

The API notebook generates its own synthetic data and does not require this download.

## Build and launch

Open PowerShell in the project folder and start Docker Desktop.

### Build the image

```powershell
docker build -t autogluon-tutorial .
```

The first build downloads and installs dependencies, so it may take several minutes.

### Launch JupyterLab

```powershell
docker run --rm -it --name autogluon-tutorial-jupyter -p 127.0.0.1:8888:8888 --mount "type=bind,source=$($PWD.Path),target=/workspace" autogluon-tutorial
```

Open the URL printed in the terminal that begins with:

```text
http://127.0.0.1:8888/lab?token=...
```

Keep the terminal open while using JupyterLab. The project folder is mounted at `/workspace` inside the container. Notebook changes and generated models are saved directly to the local project folder.

### Stop JupyterLab

Press **Ctrl+C** in the terminal and confirm shutdown.

The container is removed when it exits. Your local files and the Docker image remain available.

## Notebook walkthrough

### 1. AutoGluon API introduction

Open `autogluon.API.ipynb` and run the cells in order.

This notebook introduces:

- `TabularPredictor()` to define the prediction task.
- `.fit()` to train models.
- `.predict()` to generate target predictions.
- `.evaluate()` to measure predictive performance.
- `.leaderboard()` to compare trained models.
- `.feature_importance()` to calculate permutation importance.
- `TabularPredictor.load()` to reload a saved predictor.

At the end there is a checkpoint exercise to test the concepts taught.

### 2. NYC Airbnb example

Open `autogluon.example.ipynb` and run the cells in order.

The lessons cover:

1. Loading and understanding the dataset.
2. Cleaning the data and creating an 80/20 train/test split.
3. Training an AutoGluon predictor.
4. Generating predictions and comparing performance.
5. Interpreting feature importance.
6. Comparing AutoGluon with a Linear Regression baseline.
7. Exploring mean and median prices by borough and room type.
8. Exploring amenities mentioned in listing titles.
9. Training a new predictor with a different feature set.
10. Reviewing the methods and limitations.


## Modeling approach

The target is `price`, measured in US dollars per night.

The example keeps listings that:

- Are priced between $15 and $1,000 per night.
- Are available at least 52 days per year.
- Have a known `reviews_per_month` value.

Identifiers and selected host/date columns are removed. The remaining data is split into training and test sets using a fixed random seed.

The original AutoGluon predictor includes listing titles. A second predictor excludes titles so it can be compared with Linear Regression using the same input columns and test rows.

The Linear Regression pipeline imputes missing numeric values, scales numeric features, and one-hot encodes categorical features. These transformations are fitted only on training data.

## Evaluation and interpretation

| Metric | Meaning | Better result |
|---|---|---|
| MAE | Average absolute prediction error, in dollars. | Lower |
| RMSE | Error in dollars that gives more weight to large mistakes. | Lower |
| R² | Performance relative to predicting the test target's mean. Values can be negative. | Higher |

AutoGluon reports error-based scores such as MAE in a negative, higher-is-better form. For example, `-45` is better than `-50`. The notebook also calculates conventional positive MAE and RMSE values.

Use validation results to guide model selection and experimentation. Keep test data out of training and avoid choosing changes based on test results.

Permutation importance measures how much performance worsens when a feature is shuffled. It does not establish causation. Correlated features can substitute for each other and appear less important individually.

## Installed model dependencies

The environment includes optional dependencies for LightGBM, CatBoost, XGBoost. Random Forest and Extra Trees are also available through scikit-learn. PyTorch and FastAI dependencies are not included, so neural-network candidates may be skipped.

The models that finish training depend on the time limit, available memory, and hardware.

## Saved models and repeat runs

AutoGluon saves predictors under `AutogluonModels/`.

Use a new save path when repeating an experiment:

```python
demo_path = "AutogluonModels/tutorial_demo_run2"
```

Restarting a notebook kernel does not remove saved model files. Training at an existing path may overwrite artifacts. Do not commit generated models to Git.

## Troubleshooting

### - Dataset not found

Confirm the file is named `AB_NYC_2019.csv` and is inside the project's `data/` folder.

### - Port 8888 is already in use

Change the host port in the launch command to:

```text
-p 127.0.0.1:8889:8888
```

Then open `http://127.0.0.1:8889` and use the token printed in the terminal.

### - A model is skipped

Check the training log for missing dependencies, memory constraints, or an exhausted time limit. Training can still succeed using the remaining models.

### - Dependencies were changed

Rebuild the image, stop the existing container, and launch a new container to use the updated environment.

### - Low disk space or memory

Free disk space before further training. Close unused applications or adjust Docker/WSL resource limits.

## Limitations

- Listings are from 2019 and may not reflect current prices
- Filtering excludes some types of listings
- Title keywords do not reliably confirm that an amenity is present
- Results can vary across runs, training budgets, and computing environments.

## References

- AutoGluon documentation: https://auto.gluon.ai/stable/
- AutoGluon tabular tutorial: https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html
- NYC Airbnb dataset: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
