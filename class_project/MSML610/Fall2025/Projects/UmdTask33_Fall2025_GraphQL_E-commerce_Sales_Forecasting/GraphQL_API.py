# GraphQL_API.py
# Standalone FastAPI + GraphQL service for the e-commerce sales forecasting model

import graphene
from fastapi import FastAPI
from starlette_graphene3 import GraphQLApp, make_graphiql_handler

from GraphQL_utils import load_trained_model_and_features

# -------------------------------------------------------------------
# 1. Load trained model + lagged features (once, at startup)
# -------------------------------------------------------------------
model, lagged, feature_cols = load_trained_model_and_features()
print("Loaded model and features in GraphQL_API.py")
print("Lagged shape:", lagged.shape)
print("Feature cols:", feature_cols)


# -------------------------------------------------------------------
# 2. Helper functions that use the loaded data/model
# -------------------------------------------------------------------
def build_feature_row(shop_id: int, item_id: int, date_block_num: int):
    """
    Find the row in `lagged` for the given (shop_id, item_id, date_block_num)
    and return a 1-row DataFrame with just the feature columns.
    """
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

    # keep as DataFrame
    row = rows.iloc[[0]]
    X = row[feature_cols].copy()
    return X


def predict_sales(shop_id: int, item_id: int, date_block_num: int) -> float:
    """
    Use the RandomForest model to predict item_cnt_month.
    """
    X = build_feature_row(shop_id, item_id, date_block_num)
    y_pred = model.predict(X)[0]
    return float(y_pred)


# -------------------------------------------------------------------
# 3. GraphQL schema
# -------------------------------------------------------------------
class SalesPredictionType(graphene.ObjectType):
    shop_id = graphene.Int()
    item_id = graphene.Int()
    date_block_num = graphene.Int()
    prediction = graphene.Float()


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


schema = graphene.Schema(query=Query)


# -------------------------------------------------------------------
# 4. FastAPI app + /graphql endpoint
# -------------------------------------------------------------------
app = FastAPI(title="E-commerce Sales Forecasting API")


@app.get("/")
def root():
    return {"message": "E-commerce sales forecasting GraphQL API. Go to /graphql for GraphiQL UI."}


app.add_route(
    "/graphql",
    GraphQLApp(
        schema=schema,
        on_get=make_graphiql_handler(),  # GraphiQL UI in the browser
    ),
)
