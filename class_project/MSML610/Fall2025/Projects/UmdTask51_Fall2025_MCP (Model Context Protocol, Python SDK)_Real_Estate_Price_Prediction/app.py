import streamlit as st
import pandas as pd
import asyncio
from fastmcp import Client
import os
import json

MCP_SERVER_PATH = "./MCP_utils.py"

def load_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #020617, #0f172a);
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            color: #e5e7eb;
        }

        .card {
            background-color: #020617;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            margin-top: 1rem;
        }

        .stButton > button {
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            color: white;
            border-radius: 10px;
            height: 3em;
            font-size: 1rem;
            font-weight: 600;
            border: none;
        }

        .stButton > button:hover {
            transform: scale(1.02);
        }

        div[data-testid="metric-container"] {
            background-color: #020617;
            border-radius: 14px;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def get_production_model_async():
    """Fetches the currently designated production model ID and metadata."""
    client = Client(MCP_SERVER_PATH)
    async with client:
        return await client.call_tool("get_production_model")

async def call_mcp_predict_async(run_id, raw_features_dict):
    """Calls the prediction tool on the server."""
    client = Client(MCP_SERVER_PATH)
    async with client:
        return await client.call_tool(
            "predict_house_price",
            {"run_id": run_id, "raw_features_dict": raw_features_dict}
        )

DUMMY_RAW_FEATURES = {
    "bedrooms": 4,
    "bathrooms": 2.5,
    "sqft_living": 2500,
    "sqft_lot": 10000,
    "floors": 2,
    "waterfront": 0,
    "view": 1,
    "condition": 3,
    "grade": 8,
    "sqft_above": 1800,
    "sqft_basement": 700,
    "yr_built": 2000,
    "yr_renovated": 0,
    "zipcode": 98001,
    "lat": 47.52,
    "long": -122.31,
    "sqft_living15": 2400,
    "sqft_lot15": 9500,
    "year_sold": 2014,
    "month_sold": 6, 
    "day_of_week": 3,
}

def main():
    load_css()

    st.title("MCP based House Price Predictor")

    # --- Fetch production model silently ---
    prod_model_result = run_async(get_production_model_async())
    prod_model_data = prod_model_result.structured_content["result"]
    prod_id = prod_model_data.get("production_run_id")

    if not prod_id:
        st.error("No production model available. Please deploy a model first.")
        return

    # --- Input Form ---
    with st.form("prediction_form"):
        st.markdown("### Enter house details")

        col1, col2 = st.columns(2)

        with col1:
            bedrooms = st.number_input("Bedrooms", 1, 10, 4)
            bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.5, step=0.5)
            sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 2500)
            sqft_lot = st.number_input("Lot Size (sqft)", 1000, 50000, 10000)
            floors = st.number_input("Floors", 1, 3, 2)

        with col2:
            grade = st.slider("Grade", 1, 13, 8)
            condition = st.slider("Condition", 1, 5, 3)
            yr_built = st.number_input("Year Built", 1900, 2025, 2000)
            zipcode = st.number_input("Zipcode", 98000, 98999, 98001)
            waterfront_ui = st.radio(
                "Waterfront",
                options=["No", "Yes"],
                horizontal=True
            )

            waterfront = 1 if waterfront_ui == "Yes" else 0

        submit = st.form_submit_button("Predict Price")

    if submit:
        with st.spinner("Predicting price..."):
            raw_features = {
                **DUMMY_RAW_FEATURES,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "sqft_lot": sqft_lot,
                "floors": floors,
                "grade": grade,
                "condition": condition,
                "yr_built": yr_built,
                "zipcode": zipcode,
                "waterfront": waterfront,
            }

            result = run_async(call_mcp_predict_async(prod_id, raw_features))
            prediction = result.structured_content["result"]

        if prediction.get("status") == "SUCCESS":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(
                label="Estimated House Price",
                value=f"${prediction['predicted_price']:,.0f}",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Prediction failed. Please try again.")



if __name__ == "__main__":
    if not os.path.exists(MCP_SERVER_PATH):
        st.error(f"FATAL ERROR: MCP Server script not found at {MCP_SERVER_PATH}. Cannot start client.")
    else:
        main()