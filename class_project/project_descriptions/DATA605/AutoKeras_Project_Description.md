# Description

AutoKeras is an open-source AutoML library for deep learning, built on Keras and
TensorFlow. It solves the problem of hand-designing and tuning a neural network by
searching architectures and hyperparameters automatically for tabular, image, text,
and time-series data. It is worth a 60-minute tutorial because a few lines of code
launch a full architecture search, and the trade-off between search budget and
accuracy can be measured.

## Technologies Used

AutoKeras

- Task APIs: `StructuredDataRegressor`, `StructuredDataClassifier`,
  `ImageClassifier`, `TextClassifier`, and `TimeseriesForecaster`
- Automatic architecture search and hyperparameter tuning, controlled by `max_trials`
  and the choice of tuner
- `AutoModel` with input and head blocks to define a custom search space
- Export of the best model as a plain Keras model

# Tutorial

- Implement the tutorial "Learn AutoKeras in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the Fall2025 AutoKeras image project, and reuse what is
    good
    - `class_project/msml610/Fall2025/projects/UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras/`
  - Read the scripts of the Fall2025 AutoKeras forecasting project, which has no
    `README.md`
    - `class_project/msml610/Fall2025/projects/TutorTask_67_Fall2025_AutoKeras_Electricity_Load_Forecasting/`
  - Read the `README.md` of `tutorials/TensorFlow/` for the Keras basics
- Create `tutorials/AutoKeras/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with FLAML and AutoGluon from the AutoML point of view, e.g.,
  accuracy for the same search time
- Deliverables:
  - `autokeras_utils.py`
  - `autokeras.API.ipynb`
  - `autokeras.example.ipynb`

# Project

## Project 1: House Price Prediction with AutoML

- **Difficulty**: 1 (Easy)
- **Project Objective**: Predict house sale prices with an automatically searched
  neural network, and measure how much the search beats simple baselines
- **Dataset Suggestions**:
  [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Tasks**:
  - **Preprocess the Data**: Load the Ames data, split it into train and test, and
    report the missing values per column
  - **Define the Problem**: Predict the log of `SalePrice` as a regression problem,
    and fit a mean predictor and a `Ridge` regression as baselines
  - **Search with AutoKeras**: Fit `StructuredDataRegressor` with `max_trials` of 10
    and 30, and record the search time and the validation loss of each trial
  - **Evaluate the Model**: Compare MAE and R-squared of the best model with the
    baselines on the test set
  - **Visualize the Results**: Plot the predicted vs. actual prices, and the
    validation loss vs. the trial number
- **Bonus Ideas (Optional)**: Add engineered features, e.g., house age and total
  area, and check if the search improves; serve the exported model with Streamlit

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Data
  - Result: `tutorials/AutoKeras/` container running, and the table of missing values
    with the train and test split of the Ames data
- Milestone 2: API notebook
  - Project tasks: Search with AutoKeras
  - Result: `autokeras.API.ipynb` covering `StructuredDataRegressor`, `max_trials`,
    `AutoModel` with custom blocks, and `export_model()`
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Search with AutoKeras, Evaluate the Model,
    Visualize the Results
  - Result: `autokeras.example.ipynb` running end to end

## Project 2: Image Classification with Architecture Search

- **Difficulty**: 2 (Medium)
- **Project Objective**: Classify small color images with `ImageClassifier` and
  compare the searched network with a hand-designed CNN
- **Dataset Suggestions**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
  using a subset of 10,000 training images to fit the Docker container
- **Tasks**:
  - **Load the Images**: Load a stratified subset of CIFAR-10 and show one image per
    class
  - **Build a Baseline CNN**: Train a small Keras CNN with a fixed architecture
  - **Search the Architecture**: Run `ImageClassifier` with the `greedy` and
    `hyperband` tuners for the same number of trials
  - **Evaluate the Models**: Compare test accuracy, parameter count, and search time
    of the three models
  - **Inspect the Best Model**: Export the best model and print its layers with
    `model.summary()`
- **Bonus Ideas (Optional)**: Add data augmentation blocks to the search space

## Project 3: Weather Forecasting with a Time-Series Search

- **Difficulty**: 3 (Hard)
- **Project Objective**: Forecast the next-hour temperature with
  `TimeseriesForecaster`, and measure how stable the search result is across seeds
- **Dataset Suggestions**:
  [Jena Climate](https://www.kaggle.com/datasets/mnassrib/jena-climate)
- **Tasks**:
  - **Resample the Data**: Aggregate the 10-minute records to hourly means and build
    sliding windows of 24 hours
  - **Define the Baselines**: Compute the persistence forecast and a seasonal-naive
    forecast with a 24-hour lag
  - **Search the Forecaster**: Fit `TimeseriesForecaster` with five random seeds and
    a fixed `max_trials`
  - **Evaluate the Forecasts**: Report the mean and standard deviation of the test
    MAE and RMSE across the seeds
  - **Analyze the Trade-Offs**: Plot the MAE vs. the search time for `max_trials` of
    5, 10, and 20
- **Bonus Ideas (Optional)**: Add pressure and humidity as extra input features and
  test the gain
