# GraphQL_API – serving the model with FastAPI + GraphQL

This file is about the **serving part** of the project.  
Training happens in `GraphQL_example.ipynb`, but at some point the model has to leave the notebook and live as a real API. That is what `GraphQL_API.py` does.

There is also a small `GraphQL_API.ipynb` notebook where I played with queries and HTTP calls, but the final thing that actually runs is the Python file.
 

## 1. What `GraphQL_API.py` is doing in plain words

Short Summary

1. load the trained RandomForest model and the lagged table,
2. define a GraphQL schema with a `predictSales` query,
3. wrap that schema inside a FastAPI app,
4. start a server with uvicorn so clients can send GraphQL queries and get back JSON.

You can imagine it like this:


models/ (joblib + json)
      |
      v
GraphQL_API.py
      - load model + lagged data
      - create FastAPI app
      - put GraphQL endpoint at /graphql
      |
      v
uvicorn runs the app
      |
      v
browser / python client calls /graphql with predictSales query


## 2. Imports and loading the model

At the top of `GraphQL_API.py` we import the main libraries:

```python
import graphene
from fastapi import FastAPI
from starlette_graphene3 import GraphQLApp, make_graphiql_handler

from GraphQL_utils import load_trained_model_and_features
```

Key pieces:

* **graphene** – for defining the GraphQL schema in Python,
* **fastapi** – for creating the web app,
* **starlette-graphene3** – connects FastAPI and GraphQL together and gives us GraphiQL UI.

Then we load everything we need **once at startup**:

```python
model, lagged, feature_cols = load_trained_model_and_features()
print("Loaded model and features in GraphQL_API.py")
print("Lagged shape:", lagged.shape)
print("Feature cols:", feature_cols)
```

`load_trained_model_and_features()` (from `GraphQL_utils.py`) does most of the work

* reads `models/rf_ecommerce_monthly.joblib`,
* reads `models/model_meta.json`,
* rebuilds the monthly + lagged DataFrame from the Kaggle csv files,
* returns:

  * `model` –> trained RandomForest,
  * `lagged` –> full lagged table,
  * `feature_cols` –> list of feature columns in the right order.

So after that line runs the service has everything it needs in the memory


## 3. Building the feature row and prediction helper

To keep the GraphQL resolver clean, I use two helper functions:

```python
def build_feature_row(shop_id: int, item_id: int, date_block_num: int):
    mask = (
        (lagged["shop_id"] == shop_id)
        & (lagged["item_id"] == item_id)
        & (lagged["date_block_num"] == date_block_num)
    )
    rows = lagged.loc[mask]

    if rows.empty:
        raise ValueError(
            f"No data for shop_id={shop_id}, item_id={item_id}, date_block_num={date_block_num}"
        )

    row = rows.iloc[[0]]              # keep as 1-row DataFrame
    X = row[feature_cols].copy()      # only the columns the model expects
    return X


def predict_sales(shop_id: int, item_id: int, date_block_num: int) -> float:
    X = build_feature_row(shop_id, item_id, date_block_num)
    y_pred = model.predict(X)[0]
    return float(y_pred)
```

So the flow for one prediction is:

1. look up the right row in `lagged` by `(shop_id, item_id, date_block_num)`,
2. keep just `feature_cols`,
3. send that 1-row frame into `model.predict`,
4. turn the result into a plain Python float.

GraphQL will call `predict_sales(...)` under the hood.

## 4. Defining the GraphQL schema

Next we define how the GraphQL world looks.

### 4.1. Output type

```python
class SalesPredictionType(graphene.ObjectType):
    shop_id = graphene.Int()
    item_id = graphene.Int()
    date_block_num = graphene.Int()
    prediction = graphene.Float()
```

This is the shape of the object that `predictSales` will return.

### 4.2. Root query

```python
class Query(graphene.ObjectType):
    predict_sales = graphene.Field(
        SalesPredictionType,
        shop_id=graphene.Int(required=True),
        item_id=graphene.Int(required=True),
        date_block_num=graphene.Int(required=True),
    )

    def resolve_predict_sales(self, info, shop_id, item_id, date_block_num):
        y = predict_sales(shop_id, item_id, date_block_num)
        return SalesPredictionType(
            shop_id=shop_id,
            item_id=item_id,
            date_block_num=date_block_num,
            prediction=y,
        )
```

From GraphQL’s point of view this looks like

```graphql
type SalesPrediction {
  shopId: Int!
  itemId: Int!
  dateBlockNum: Int!
  prediction: Float!
}

type Query {
  predictSales(shopId: Int!, itemId: Int!, dateBlockNum: Int!): SalesPrediction!
}
```

The resolver `resolve_predict_sales` is where Python and GraphQL meet:

* GraphQL sends in `shopId`, `itemId`, `dateBlockNum`,
* Python receives them as `shop_id`, `item_id`, `date_block_num`,
* the function calls `predict_sales(...)`,
* and returns a `SalesPredictionType` instance.

Finally we build the schema:

```python
schema = graphene.Schema(query=Query)
```

## 5. FastAPI app and /graphql route

Now we make the web app:

```python
app = FastAPI(title="E-commerce Sales Forecasting API")
```

A tiny root endpoint for quick check

```python
@app.get("/")
def root():
    return {
        "message": "E-commerce sales forecasting GraphQL API. "
                   "Go to /graphql for GraphiQL UI."
    }
```

Then the important part: mount the GraphQL app at `/graphql`:

```python
app.add_route(
    "/graphql",
    GraphQLApp(
        schema=schema,
        on_get=make_graphiql_handler(),  # this gives the nice GraphiQL web UI
    ),
)
```

So:

* `GET /graphql`
  returns the GraphiQL HTML page (a little React app).
* `POST /graphql`
  executes a GraphQL query and returns JSON.

This is all we need for the serving layer.


## 6. How to run the API locally

From the project root, inside your virtual environment:

```bash
uvicorn GraphQL_API:app --reload --port 8000
```

You should see logs like:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Now:

1. open the root endpoint:
   [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   You should see a small JSON message with a hint to go to `/graphql`.

2. open the GraphQL playground:
   [http://127.0.0.1:8000/graphql](http://127.0.0.1:8000/graphql)

   In the left panel of GraphiQL, paste:

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

   Then click the **Play** button.
   On the right you will see a JSON response similar to:

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

## 7. Calling the API from Python (GraphQL client)

If you don’t want to use the browser, you can call the same endpoint from a a  notebook or script:

```python
import requests
import json

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

This is the same query as before, just wrapped in a Python string and sent as JSON. It is nice for unit tests or integraion tests later.


## 8. How this fits with Docker

The **Dockerfile** uses `GraphQL_API.py` as the entrypoint command:

```dockerfile
CMD ["uvicorn", "GraphQL_API:app", "--host", "0.0.0.0", "--port", "8000"]
```

So when the container starts, it basically runs the exact same uvicorn command you use localy, just inside the image.

From the project root:

```bash
docker build -t ecommerce-graphql .

docker run --rm -p 8000:8000 ecommerce-graphql
```

Then again you hit:

* [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* [http://127.0.0.1:8000/graphql](http://127.0.0.1:8000/graphql)

The behaviour is identical, only difference is that now it is fully self contained inside Docker.


## 9. Summary

So `GraphQL_API.py` is the “production” face of the project:

* it loads the trained model and lagged data,
* it defines a clean GraphQL schema with `predictSales`,
* it exposes everything through FastAPI at `/graphql`,
* it is ready to run both on your machine and in a Docker container.


