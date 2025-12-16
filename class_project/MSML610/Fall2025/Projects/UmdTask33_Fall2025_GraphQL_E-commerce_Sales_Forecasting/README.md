Mihir Chotneeru 

UID : 121301635

# MSML610 - E-commerce Monthly Sales Forecasting with GraphQL (Difficulty : Hard)
given a shop, an item and a month, we predict how many units will be sold in that month**.  
Then we put that prediction behind a **GraphQL API** and also make it run inside a **Docker** container.

The work is split into three main parts:

1. build a clean monthly dataset from the raw Kaggle sales data  
2. train and evaluate a machine learning model for `item_cnt_month`  
3. serve the trained model through a FastAPI + GraphQL endpoint


## 1. Repository layout

Rough structure of the repo:


.
├── data/                        # raw Kaggle csv files (local, not committed)
├── models/
│   ├── rf_ecommerce_monthly.joblib   # trained RandomForest
│   └── model_meta.json               # feature list, lags, etc
│
├── GraphQL_example.ipynb        # main modelling notebook
├── GraphQL_example.md           # short text description of the notebook
│
├── GraphQL_API.ipynb            # small playground for GraphQL ideas
├── GraphQL_API.py               # final FastAPI + GraphQL service
├── GraphQL_API.md               # explanation of the API and schema
│
├── GraphQL_utils.py             # helpers for data, lags, training, loading
├── utils_data_io.py             # provided helper for reading data
├── utils_post_processing.py     # provided helper file
│
├── Dockerfile                   # container recipe for the API service
├── requirements.txt             # python dependencies
└── README.md
````

So the notebook "GraphQL_example.ipynb" does the modelling,
"GraphQL_utils.py" holds common functions,
and "GraphQL_API.py" is the file that uvicorn runs to expose GraphQL.

---

### 1. Workflow (step-by-step)

[1] Raw Kaggle CSV files (sales_train, items, shops, etc)
      |
      v
[2] utils_data_io.load_raw_kaggle_data()
      |
      v
[3] GraphQL_utils.build_base_training_table()
      - aggregate to monthly level
      - compute item_cnt_month and avg_item_price
      - add year and month columns
      |
      v
[4] GraphQL_utils.make_lagged_features(lags = 1, 2, 3)
      - create lag_1, lag_2, lag_3 from item_cnt_month
      |
      v
[5] GraphQL_utils.make_train_val_sets()
      - split data into train months vs last validation month
      |
      v
[6] Train RandomForest in GraphQL_example.ipynb
      - fit model on training months
      - evaluate MAE / RMSE
      - compare vs lag-1 naive baseline
      |
      v
[7] Train final model on all data
      - save rf_ecommerce_monthly.joblib
      - save model_meta.json (feature_cols, lags)
      |
      v
[8] GraphQL_API.py startup
      - load_trained_model_and_features()
      - keep model, lagged_df, feature_cols in memory
      |
      v
[9] FastAPI creates / and /graphql endpoints
      |
      v
[10] Client (browser GraphiQL, Python requests, etc)
       sends a predictSales GraphQL query:
       (shopId, itemId, dateBlockNum)
      |
      v
[11] GraphQL resolver calls predict_sales()
       - find matching row in lagged_df
       - build 1-row feature dataframe
       - run model.predict()
      |
      v
[12] FastAPI / GraphQL returns JSON with prediction
       - shopId, itemId, dateBlockNum, prediction
      |
      v
[13] Docker
       - Dockerfile builds image with code + model
       - docker run -p 8000:8000 ecommerce-graphql
       - same /graphql endpoint, just running in container

## 2. Data and feature engineering

The dataset is the classic Kaggle **Predict Future Sales** data
Each row in the raw file is a single day sale:

* `date` – day as string
* `date_block_num` – month index (0, 1, 2, …)
* `shop_id`
* `item_id`
* `item_price`
* `item_cnt_day` – how many units sold that day

We turn this into a monthly table per `(shop_id, item_id)` pair.

Steps (implemented in `GraphQL_utils.py` and used in the notebook):

1. parse `date` into proper datetime
2. group by `(date_block_num, shop_id, item_id)`

   * sum of `item_cnt_day` → `item_cnt_month`
   * mean of `item_price` → `avg_item_price`
3. create calender columns from the date

   * `year`
   * `month`
4. create **lag features** of the target:

   * `lag_1`, `lag_2`, `lag_3` = `item_cnt_month` shifted by 1, 2, 3 months
   * rows where any lag is missing are dropped

Final set of features used to train

["shop_id", "item_id", "avg_item_price", "year", "month", "lag_1", "lag_2", "lag_3"]

Target variable is:

"item_cnt_month"

This same feature list gets writen to `models/model_meta.json` so the API knows
exactly which columns and order to use later.


## 3. Modelling part (GraphQL_example.ipynb)

The notebook follows a natural order:

1. **load data** using the helper functions

2. **build the monthly table** with `build_base_training_table(sales)`

3. **add lagged features** with `make_lagged_features(monthly, lags=(1, 2, 3))`

4. **make train and validaton sets**

   * most months go into training
   * the last month becomes the validation month
   * done by `make_train_val_sets(lagged_df)`

5. **train a RandomForestRegressor**

   Example settings (the real values are printed in the notebook):

   * `n_estimators` around 80–100
   * `max_depth` around 10–12
   * `random_state = 42`

6. **evaluate the model**

   Main metrics we check:

   * **MAE** – mean absolute error
   * **RMSE** – root mean squared error (primary metric for the project)
   * **MAPE** – checked mostly for curiosity, but can be inaccurate for tiny denominators

   We also compare against a very simple **lag-1 baseline**:

   * for each row in the validation month, baseline prediction is just `lag_1`
   * we compute RMSE for this naive method and for RandomForest

   In a typical run we can see something like:

   ```text
   MAE  ≈ 1.5
   RMSE ≈ 16.9
   MAPE ≈ a very big number because of near zero sales cases
   ```

   The important thing is RMSE of RandomForest vs RMSE of the lag-1 baseline.
   The model improves on the naive one, which is the check we care about.

7. **train final model on all data**

   After we are happy with validation performance, we rebuild `X_full` and `y_full`
   using all all allowed months, and train a slightly bigger forest for stability.
   This becomes the **production model**.

8. **save model and metadata**

   * `models/rf_ecommerce_monthly.joblib` – RandomForest
   * `models/model_meta.json` – includes `feature_cols` and `lags`

9. **sanity check prediction**

   At the end of the notebook we pick a real combination from the lagged table,
   build a 1-row feature frame, run it through the model, and print the predicted
   `item_cnt_month`. This is basically the same logic the API will later call.

---

## 4. GraphQL API idea

Once the model and metadata exist in `models/`,
`GraphQL_API.py` can load them and turn the model into a web service.

Very quick summary (the details are in `GraphQL_API.md`):

* we import `graphene`, `fastapi`, `starlette_graphene3`
* we call `load_trained_model_and_features()`
  this returns:

  * the trained RandomForest
  * the full lagged DataFrame
  * the list of feature columns
* we define a helper `predict_sales(shop_id, item_id, date_block_num)`
  that:

  * finds the matching row in the lagged DataFrame,
  * builds a 1-row feature DataFrame using the same column order,
  * calls `model.predict` and returns the float

GraphQL schema (conceptually) looks like:

```graphql
type SalesPrediction {
  shopId: Int!
  itemId: Int!
  dateBlockNum: Int!
  prediction: Float!
}

type Query {
  predictSales(
    shopId: Int!,
    itemId: Int!,
    dateBlockNum: Int!
  ): SalesPrediction!
}
```

FastAPI part:  

* `GET /` → simple JSON message saying where to go
* `GET /graphql` → GraphiQL UI in the browser
* `POST /graphql` → executes query and returns JSON

So a typical query the client sends is:

```graphql
query {
  predictSales(shopId: 0, itemId: 30, dateBlockNum: 1) {
    shopId
    itemId
    dateBlockNum
    prediction
  }
}
```

And the JSON response looks like:

```json
{
  "data": {
    "predictSales": {
      "shopId": 0,
      "itemId": 30,
      "dateBlockNum": 1,
      "prediction": 3.41
    }
  }
}
```

This is the main “contract” of the API.

---

## 5. How to run everything

### 5.1. Python environment

From the project root:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Make sure the Kaggle csv files are under `data/`
(or adjust paths in `GraphQL_utils.py` if they live somewhere else).

Then open **GraphQL_example.ipynb** and run the notebook top to bottom.
That will:

* read the raw data
* create the monthly + lagged table
* train and evaluate the RandomForest
* write `rf_ecommerce_monthly.joblib` and `model_meta.json` into `models/`

After that the API has everything it needs.

---

### 5.2. Run GraphQL API without Docker

From the project root, with the virtual env active:

```bash
uvicorn GraphQL_API:app --reload --port 8000
```

Now:

* open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) → small JSON welcome message
* open [http://127.0.0.1:8000/graphql](http://127.0.0.1:8000/graphql) → GraphiQL page

Paste the query:

```graphql
query {
  predictSales(shopId: 0, itemId: 30, dateBlockNum: 1) {
    shopId
    itemId
    dateBlockNum
    prediction
  }
}
```

You should see a prediction number on the right side.

A quick python client to hit the running API:

```python
import requests, json

url = "http://127.0.0.1:8000/graphql"
query = """
{
  predictSales(shopId: 0, itemId: 30, dateBlockNum: 1) {
    shopId
    itemId
    dateBlockNum
    prediction
  }
}
"""

response = requests.post(url, json={"query": query})
print(response.status_code)
print(json.dumps(response.json(), indent=2))
```

---

### 5.3. Run everything inside Docker

The Dockerfile builds a small image that contains:

* python base image
* project files
* installed requirements
* uvicorn command as the container entrypoint

From the project root:

```bash
# build the image
docker build -t ecommerce-graphql .

# run container in foreground
docker run --rm -p 8000:8000 ecommerce-graphql
```

Then again visit [http://127.0.0.1:8000/graphql](http://127.0.0.1:8000/graphql) and send the same query.
The experience is the same, just now it is running fully inside Docker.

If you want detached mode:

```bash
docker run -d -p 8000:8000 --name ecommerce-graphql-container ecommerce-graphql
# later to stop
docker stop ecommerce-graphql-container
```

---

## 6. What to look at when reading the project

If you are going through the repo in a hurry, a nice order is:

1. **GraphQL_example.ipynb** for the data and modelling story
2. **GraphQL_utils.py** for the helper functions and pipeline code
3. **GraphQL_API.py** to see how the model loads and how GraphQL is set up
4. **Dockerfile** just to confirm the app can be containerized and shipped

Together these snippets show a full flow:
from raw daily transactions → to monthly features and model → to a GraphQL API
that returns predictions for a given shop, item, and month.


I also added a small shared inference helper (predict_sales in GraphQL_utils.py) that both the notebook and the GraphQL API use.
This helper takes (shop_id, item_id, date_block_num) and returns the model’s forecasted item_cnt_month.
For example, for (date_block_num = 1, shop_id = 0, item_id = 30) the model predicts about 2.39 units.
The same call works from the Jupyter notebook and from the /graphql endpoint, so the logic is not duplicated.