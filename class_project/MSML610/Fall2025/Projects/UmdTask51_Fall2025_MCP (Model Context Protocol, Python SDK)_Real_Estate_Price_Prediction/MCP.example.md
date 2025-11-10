# Example: Real Estate Price Prediction

This notebook tells the story of our project: using the King County dataset to predict house prices.

Per course guidelines, this notebook is kept clean. All the "heavy lifting" (code) is moved into the `.py` utility files. This notebook just calls those functions.

## 1. Objective

Predict the sale price of houses in King County, USA, using features like `bedrooms`, `sqft_living`, etc.

## 2. Load Data

First, we load the `kc_house_data.csv` file using a helper function from `utils_data_io.py`.

## 3. Clean and Engineer Features

Next, we process the data using functions from `utils_post_processing.py`. This includes:
* Handling missing values.
* Log-transforming the `price` to handle its skew.
* Creating new features (e.g., `age_of_house`).
* Splitting the data into training and testing sets.

## 4. Train Initial Model

We now train our first XGBoost model. We call the `train_model` function from `MCP_utils.py`. This function:
1.  Wraps the training run in an `mcp.Context`.
2.  Logs our chosen hyperparameters.
3.  Trains the model.
4.  Logs the resulting test MSE metric.
5.  Logs the model file as an artifact.

## 5. Hyperparameter Tune Model

Finally, we try to improve our model. We call the `tune_model` function from `MCP_utils.py`. This function:
1.  Defines a `param_grid` for XGBoost.
2.  Wraps a `GridSearchCV` in an `mcp.Context`.
3.  Finds the best parameters and logs them.
4.  Logs the best cross-validation score.

## 6. Conclusion

We review the metrics from our tuned model to see if our performance improved.