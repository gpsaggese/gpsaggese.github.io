# GraphQL_example.ipynb – model training and experiments

This notebook is like the **lab** part of the project.  
Here I sit with the Kaggle csv files, clean them a bit, make monthly features, train the RandomForest model, and finally save the model artifacts that later the GraphQL API is using.

I’m writing this more like notes to myself, so it reads a bit casual.

## 1. Goal of this notebook

Very simple idea:

> take raw daily sales from Kaggle → turn into **monthly shop–item time series** →  
> add some lag features → train a model that predicts `item_cnt_month` for a future month.

This notebook does **three main things**:

1. build the monthly + lagged dataset,  
2. train + evaluate the RandomForest vs a naive baseline,  
3. train a final model on all data and save it to disk (`joblib` + metadata json).


---

## 2. Loading the raw Kaggle data

At the top I use the helper from `utils_data_io.py`:

- `load_raw_kaggle_data()`  
  - reads `sales_train.csv`, `items.csv`, `shops.csv`, `item_categories.csv`, `test.csv`.
  - everything is stored in a small dict called `raw_data`.

In the notebook I quickly check:

- dtypes of `sales_train`,
- date range (2013-01 → 2015-10),
- number of unique shops and items.

This is mostly sanity checking for *“ok, data is what I expect, nothing totally broken”*.



## 3. Building the monthly training table

Next I move from daily records to monthly level using `GraphQL_utils.py`:

- `build_base_training_table(sales_df)`  
  - groups by `date_block_num`, `shop_id`, `item_id`,
  - computes:
    - `item_cnt_month` = sum of daily item counts,
    - `avg_item_price` = mean of item price in that month,
  - adds `year` and `month` columns derived from `date_block_num`.

The result is a dataframe called **`monthly`**.

Quick checks in the notebook:

- `monthly.shape` – should be around 1.6M rows,  
- range of `date_block_num` (0 → 33),  
- basic stats of `item_cnt_month` (most values are small, some spikes).



## 4. Making lag features

Time series need history, so I create lagged features:

- `make_lagged_features(monthly, lags=(1, 2, 3))`

This function:

1. shifts `item_cnt_month` by 1, 2, 3 months within each `(shop_id, item_id)` series,
2. joins those shifted columns back as:
   - `lag_1`
   - `lag_2`
   - `lag_3`
3. drops rows where some lag is missing (the very first months of each series).

Result is a big `lagged` dataframe with columns:

- `['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'avg_item_price', 'year', 'month', 'lag_1', 'lag_2', 'lag_3']`  

I print `lagged.shape` to be sure it matches expectations.



## 5. Train / validation split

To measure how good the model generalizes, I hold out the **last month** as validation:

- `make_train_val_sets(lagged)` returns:
  - `X_train, y_train` – all months except the last one,
  - `X_val,   y_val`   – data for final `date_block_num` (e.g. month 33),
  - `X_val_baseline, y_val_baseline` – version needed for the naive baseline.

Features used:

```text
feature_cols = ['shop_id', 'item_id', 'avg_item_price',
                'year', 'month', 'lag_1', 'lag_2', 'lag_3']
```

Target:

* `y = item_cnt_month` (how many items we sold that month).


## 6. Baseline: lag-1 naive forecast

Before going fancy, I compare with a very dumb baseline:
**tomorrow = today**, or in our case **this month’s sales = last month’s sales**.

In the notebook:

* `y_val_baseline` = true counts on validation month
* `y_pred_naive`   = `lag_1` values on that month

Then I compute baseline RMSE (root mean squared error).
This gives me a number to beat. If the RandomForest can’t do better than this, then something is wrong.


## 7. Training the RandomForest model

For the actual model I use a `RandomForestRegressor` (from scikit-learn):

* wrapped in a helper: `train_baseline_model(X_train, y_train, n_estimators=80, max_depth=10)`
* no hyper-parameter search here, just reasonably safe values:

  * `n_estimators ≈ 80` trees,
  * `max_depth ≈ 10`,
  * random_state fixed for reproducibility.

After training, I evaluate with:

* `evaluate_model(model, X_val, y_val)` → RMSE on validation month.

In the notebook I also compute extra metrics:

* `MAE` = mean absolute error,
* `RMSE` again,
* `MAPE` (percent error).

Values I got in this run (approx):

* **MAE  ≈ 1.52**
* **RMSE ≈ 16.9**
* **Naive lag-1 RMSE ≈ 16.88** (so basically same ballpark)
* **MAPE is huge** because there are lots of rows where `y_val` is 0 or close to 0, so dividing by that explodes the percentage. I keep it mostly to show that MAPE is often a bad metric when many targets are zero.

Main takeaway:
RandomForest is doing roughly similar to the naive lag-1 baseline, that’s actually very normal for this Kaggle dataset, because last month’s demand is already a pretty strong signal. At least the model is not worse, which means the pipeline is consistent.

---

## 8. Training the final model on all data

Once I’m happy the pipeline works and metrics are not crazy, I train a **final model**:

* I call `train_baseline_model` again but on `X_full`, `y_full`,
  where `X_full` = all rows in `lagged` (no validation split).
* I slightly bump the tree count (`n_estimators=100`) and maybe max depth for more stability.

This final model is the one we deploy behind GraphQL.

I also build the **metadata**:

* list of feature columns: `feature_cols`,
* list of lags used: `[1, 2, 3]`.

Then I save everything:

* `models/rf_ecommerce_monthly.joblib` – the trained RandomForest,
* `models/model_meta.json` – small JSON with:

  ```json
  {
    "feature_cols": ["shop_id", "item_id", "avg_item_price", "year", "month",
                     "lag_1", "lag_2", "lag_3"],
    "lags": [1, 2, 3]
  }
  ```

Later, `GraphQL_utils.load_trained_model_and_features()` uses exactly these two files to:

1. load the model,
2. rebuild the `monthly` + `lagged` table the same way,
3. return `(model, lagged_df, feature_cols)` to `GraphQL_API.py`.

---

## 9. How this notebook connects to the rest of the project

Very short summary of who uses what:

* **This notebook**

  * does the data prep and model training,
  * creates and saves the artifacts.

* **GraphQL_API.py**

  * loads those artifacts,
  * exposes a GraphQL endpoint `predictSales(shopId, itemId, dateBlockNum)`.

* **Dockerfile**

  * wraps everything into a container so the same API can run anywhere.

So if the notebook runs end-to-end without errors and saves the model files, then the API + Docker part can stand on top of it quite nicely.

