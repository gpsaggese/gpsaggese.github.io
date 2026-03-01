import pandas as pd
import numpy as np

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]

    # trim extreme outliers
    for col in ["Quantity", "UnitPrice", "LineTotal"]:
        q1, q3 = df[col].quantile([0.01, 0.99])
        df = df[(df[col] >= q1) & (df[col] <= q3)]

    return df

def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    grp = df.groupby("CustomerID")

    cust = pd.DataFrame({
        "num_invoices": grp["InvoiceNo"].nunique(),
        "num_items": grp["Quantity"].sum(),
        "unique_items": grp["StockCode"].nunique(),
        "total_spend": grp["LineTotal"].sum(),
        "avg_unit_price": grp["UnitPrice"].mean(),
        "avg_basket_size": grp["Quantity"].mean(),
        "first_purchase": grp["InvoiceDate"].min(),
        "last_purchase": grp["InvoiceDate"].max()
    }).reset_index()

    cust["recency_days"] = (ref_date - cust["last_purchase"]).dt.days
    cust["tenure_days"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days.clip(lower=0)
    cust["frequency"] = cust["num_invoices"]

    ip = (df.sort_values(["CustomerID", "InvoiceDate"])
            .groupby("CustomerID")["InvoiceDate"]
            .apply(lambda s: s.diff().dt.days.dropna().mean() if s.size > 1 else np.nan))
    cust = cust.merge(ip.rename("mean_interpurchase_days"), on="CustomerID", how="left")
    cust["mean_interpurchase_days"] = cust["mean_interpurchase_days"].fillna(
        cust["mean_interpurchase_days"].median()
    )

    return cust

def feature_matrix(cust: pd.DataFrame):
    feature_cols = [
        "num_invoices","num_items","unique_items","avg_unit_price","avg_basket_size",
        "recency_days","tenure_days","frequency","mean_interpurchase_days"
    ]
    X = cust[feature_cols].copy()
    return X, feature_cols
