import pandas as pd
import numpy as np
import os
import json
import joblib
import import_ipynb
import time
import pickle # For saving model artifact
import uuid   # For generating run_id
from datetime import datetime
from typing import Dict, Any, List

# Import the notebook file containing the data and model contexts
import MCP_API 
from mcp.server.fastmcp import FastMCP

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor


# --- Initialization ---

mcp = FastMCP("RealEstatePricePredictionProject")

# Initialize the persistent and in-memory contexts
data_context = MCP_API.DataContext()
experiment_registry = MCP_API.ExperimentRegistry()
production_registry = MCP_API.ProductionRegistry()
@mcp.tool()
def upload_file(path: str) -> str:
    """
    This function read the csv data and stores it in the class variable.

    Args:
        Absolute path to the .csv file.

    Returns:
        String which shows the shape of the data.
    """

    print(f"Uploading file from path: {path}")
    if not os.path.exists(path):
        return f"Error: The file at '{path}' does not exist."

    # Check if file has a .csv extension
    if not path.lower().endswith('.csv'):
        return "Error: The file must be a CSV file."

    try:
        # Try to read the CSV file using pandas
        data = pd.read_csv(path)
        
        # Store the data in the DataContext class
        data_context.set_data(data)

        # Store the shape of the data (rows, columns)
        data_shape = data_context.get_data().shape

        return f"Data successfully loaded. Shape: {data_shape}"
    except Exception as e:
        return f"An unexpected error occured: {str(e)}"
    
@mcp.tool()
def get_columns_info() -> str:
    """
    This function gives information about columns.

    Returns:
        String which contains column names.
    """

    columns = data_context.get_data().columns

    return ", ".join(columns)

@mcp.tool()
def download_data() -> str:
    """
    Retrieves the entire stored DataFrame as a JSON string for client-side processing.

    Returns:
        String containing the JSON representation of the DataFrame.
    """
    data = data_context.get_data()
    
    if data is None:
        return "Error: No data has been uploaded yet."
        
    # **FIX: Explicitly convert to JSON using the 'records' orientation.**
    # This creates a reliable list of dictionaries: [{"col1": val, "col2": val}, ...]
    return data.to_json(orient='records')

@mcp.tool()
def engineer_features() -> str:
    """
    Performs feature engineering by extracting year, month, and day_of_week 
    from the 'date' column and then drops the original 'date' column.

    Returns:
        A confirmation string indicating the success or failure of the operation.
    """
    data = data_context.get_data()

    if data is None:
        return "Error: No data has been uploaded yet. Please run 'upload_file' first."

    if 'date' not in data.columns:
        return "Error: The 'date' column required for feature engineering was not found."

    try:
        # Convert the 'date' column to datetime objects
        # We use errors='coerce' to turn invalid parsing into NaT (Not a Time)
        data['date'] = pd.to_datetime(data['date'])
        
        # Check if the conversion resulted in mostly NaT, which could happen 
        # if the column was already processed or contained non-date data.
        if data['date'].isnull().all():
            return "Error: 'date' column could not be converted to datetime objects."
            
        # Extract new features
        data['year_sold'] = data['date'].dt.year
        data['month_sold'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        
        # Drop the original 'date' column
        data = data.drop(columns=['date'])


        def categorize_renovation_year_numeric(year):
            if year == 0:
                return 0  # Not Renovated
            elif year < 1980:
                return 1  # Before 1980
            elif year < 1990:
                return 2  # 1980-1989
            elif year < 2000:
                return 3  # 1990-1999
            elif year < 2010:
                return 4  # 2000-2009
            else:
                return 5  # 2010+


        data['was_renovated'] = (data['yr_renovated'] > 0).astype(int)
        data['renovation_period'] = data['yr_renovated'].apply(categorize_renovation_year_numeric)




        # ==============================
        # 1. Living-area related features
        # ==============================

        # Total usable square footage: above-ground + basement
        data['total_sqft'] = data['sqft_living'] + data['sqft_basement']

        # How intensively the lot is used (protect against division by zero)
        data['living_to_lot_ratio'] = (data['sqft_lot'] / data['sqft_living'])

        # Bathrooms per bedroom as a simple comfort indicator
        data['bath_per_bed'] = data['bathrooms'] / (data['bedrooms'])

        # Difference between house size and neighborhood average size
        data['living15_diff'] = data['sqft_living'] - data['sqft_living15']

        # Share of total area that is basement
        data['basement_share'] = data['sqft_basement'] / (
            data['sqft_living'] + data['sqft_basement'] + 1
        )

        # Binary flag: does the house have a basement?
        data['has_basement'] = (data['sqft_basement'] > 0).astype(int)

        # Log transforms for skewed variables (add 1 to avoid log(0))
        data['log_price'] = np.log1p(data['price'])
        data['log_sqft_living'] = np.log1p(data['sqft_living'])
        data['log_sqft_lot'] = np.log1p(data['sqft_lot'])


        # Neighborhood / location: density / land use
        data['lot_per_living'] = data['sqft_lot'] / data['sqft_living']

        # ==========================================
        # 3. Age, condition, renovation-related features
        # ==========================================

        # House age at time of sale
        data['house_age'] = data['year_sold'] - data['yr_built']

        # Years since last renovation (or age if never renovated)
        data['since_renovation'] = np.where(
            data['yr_renovated'] > 0,
            data['year_sold'] - data['yr_renovated'],
            data['house_age']
        )

        # Simple combined quality score (condition + grade)
        data['quality_score'] = data['condition'] + data['grade']
        
        # Store the modified DataFrame back in the context
        data_context.set_fe_data(data)

        return f"Feature Engineering Successful. Shape: {data.shape}"

    except Exception as e:
        return f"Feature Engineering Error: {str(e)}"
    
@mcp.tool()
def add_features() -> str:
    """
    Performs feature engineering by extracting year, month, and day_of_week 
    from the 'date' column and then drops the original 'date' column.

    Returns:
        A confirmation string indicating the success or failure of the operation.
    """
    data = data_context.get_fe_data()

    if data is None:
        return "Error: No data has been uploaded yet. Please run 'upload_file' first."

    try:
        # total_sqft: safe
        data['total_sqft'] = data['sqft_living'] + data['sqft_basement']

        # living_to_lot_ratio = sqft_living / sqft_lot
        data['living_to_lot_ratio'] = np.where(
            data['sqft_lot'] != 0,
            data['sqft_living'] / data['sqft_lot'],
            0.0    
        )

        # bath_per_bed = bathrooms / bedrooms
        data['bath_per_bed'] = np.where(
            data['bedrooms'] != 0,
            data['bathrooms'] / data['bedrooms'],
            0.0
        )

        # living15_diff: safe
        data['living15_diff'] = data['sqft_living'] - data['sqft_living15']

        # basement_share = sqft_basement / (sqft_living + sqft_basement)
        denom = data['sqft_living'] + data['sqft_basement']
        data['basement_share'] = np.where(
            denom != 0,
            data['sqft_basement'] / denom,
            0.0
        )

        # has_basement: safe
        data['has_basement'] = (data['sqft_basement'] > 0).astype(int)
        
        # Store the modified DataFrame back in the context
        data_context.set_fe_data(data)

        return "Finished Feature Engineering"

    except Exception as e:
        return f"Feature Engineering Error: {str(e)}"
    


@mcp.tool()
def download_fe_data() -> str:
    """
    Retrieves the entire stored DataFrame as a JSON string for client-side processing.

    Returns:
        String containing the JSON representation of the DataFrame.
    """
    data = data_context.get_fe_data()
    
    if data is None:
        return "Error: No feature engineered data has been generated yet."

    # **FIX: Explicitly convert to JSON using the 'records' orientation.**
    # This creates a reliable list of dictionaries: [{"col1": val, "col2": val}, ...]
    return data.to_json(orient='records')


def _get_model_artifact(model_name: str, **hyperparameters) -> Any:
    """
    Initializes the specific model based on its name and hyperparameters.
    """
    if "LinearRegression" in model_name:
        # LinearRegression takes no hyperparameters usually, but we pass them just in case
        return LinearRegression()
    
    elif "ScaledLinearRegression" in model_name:
        # Create a pipeline with scaling followed by linear regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        return pipeline
    
    elif "Ridge" in model_name:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        return pipeline
    
    elif "Lasso" in model_name:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(max_iter=10000))
        ])
        return pipeline
    elif "RandomForest" in model_name:
        return RandomForestRegressor(**hyperparameters)
    elif "XGBoost" in model_name:
        from xgboost import XGBRegressor
        return XGBRegressor(**hyperparameters)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# --- Core Training Tool ---

@mcp.tool()
def run_experiment(mod_config: Dict[str, Any], train_config: Dict[str, Any], feature_list: List[str]) -> Dict[str, Any]:
    """
    Orchestrates a complete training run, persists the model artifact, 
    registers metadata, and returns a run summary.
    """
    start_time = time.time()
    
    # 1. Load Data and Configuration
    fe_data = data_context.get_fe_data()
    if fe_data is None:
        return {"error": "No Feature Engineered data available for training."}
    
    try:
        # Convert input dictionaries to immutable dataclasses for tracking
        try:
            model_cfg = MCP_API.ModelConfig(**mod_config)
        except Exception as e:
            raise ValueError(f"Invalid model configuration: {str(e)}")
        
        try:
            train_cfg = MCP_API.TrainingConfig(**train_config)
        except Exception as e:
            raise ValueError(f"Invalid training configuration: {str(e)}")
             
        try:
            X = fe_data[feature_list]
        except KeyError as e:
            raise ValueError(f"Feature list contains invalid column names: {e}")

        y = fe_data["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_cfg.test_size, random_state=train_cfg.random_state
        )

        # 3. Model Initialization and Training
        try:
            model_artifact = _get_model_artifact(model_cfg.model_name, **model_cfg.hyperparameters)
        except ValueError as e:
            raise ValueError(f"Model initialization error: {str(e)}")
        
        if 'Ridge' or 'Lasso' in model_cfg.model_name:
            grid = GridSearchCV(
                model_artifact,
                model_cfg.hyperparameters,
                cv=5,
                scoring='r2',
                return_train_score=True,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            model_artifact = grid.best_estimator_
        else:
            try:
                model_artifact.fit(X_train, y_train)
            except Exception as e:
                raise RuntimeError(f"Model training failed: {str(e)}")
        
        train_time_sec = time.time() - start_time

        # 4. Metric Computation
        y_pred = model_artifact.predict(X_test)
        y_train_pred = model_artifact.predict(X_train)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        price_mean = y_test.mean()
        price_range = y_test.max() - y_test.min()
        
        metrics = MCP_API.Metrics(
            train_r2=r2_score(y_train, y_train_pred),
            train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred)),
            test_r2=r2,
            test_rmse=rmse,
            test_rmse_pct_mean=rmse / price_mean * 100,
            test_rmse_pct_range=rmse / price_range * 100,
            train_time_sec=train_time_sec,
            best_param=grid.best_params_ if model_cfg.hyperparameters else None
        )

        # 5. Artifact Persistence (Model Object)
        run_id = str(uuid.uuid4())
        artifact_filename = f"{run_id}.pkl"
        artifact_path = os.path.join(experiment_registry.REGISTRY_DIR, artifact_filename)
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(model_artifact, f)

        # 6. Metadata Registration
        record = MCP_API.ExperimentRecord(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            model_config=model_cfg,
            training_config=train_cfg,
            metrics=metrics,
            artifact_path=artifact_path,
            features_used=feature_list
        )
        experiment_registry.append_record(record)

        # 7. Return Run Summary
        summary = experiment_registry.get_run_summary(run_id)
        if summary:
            summary['status'] = "SUCCESS"
            return summary
        else:
            return {"error": "Failed to retrieve run summary after registration."}
        
    except Exception as e:
        return {"error": f"Experiment run failed: {str(e)}"}
    
@mcp.tool()
def get_model_coefficients(run_id: str) -> Dict[str, Any] | str:
    """
    Loads the model artifact for a specific run, extracts its coefficients 
    and feature names, and returns them to the client.
    """
    
    # 1. Retrieve the FULL Experiment Record
    # We load the full registry to access the 'artifact_path' and 'features_used' reliably.
    full_runs = experiment_registry.list_runs()
    full_record = next((r for r in full_runs if r['run_id'] == run_id), None)
    
    if not full_record:
        return f"Error: Run ID '{run_id}' not found in the registry."

    artifact_path = full_record.get('artifact_path')
    model_name = full_record['model_config']['model_name']
    
    if not os.path.exists(artifact_path):
        return f"Error: Model artifact not found at the recorded path: {artifact_path}"
        
    # 2. Load and Extract the Final Estimator
    try:
        model = None
        
        # Load the artifact (which could be the model or a Pipeline)
        with open(artifact_path, 'rb') as f:
            artifact = pickle.load(f)
            
        # Determine the final model object based on the model name
        if 'LinearRegression' in model_name:
            # Simple model, stored directly
            model = artifact
            
        elif 'ScaledLinearRegression' in model_name:
            # Pipeline model: Must extract the final regressor step
            if hasattr(artifact, 'named_steps') and 'regressor' in artifact.named_steps:
                 model = artifact.named_steps['regressor']
            else:
                 # This error means the pipeline structure is not what's expected
                 return f"Error: Could not find 'regressor' step in the pipeline for {model_name}. Check the step name in your training code."
        elif 'Ridge' in model_name:
            # Pipeline model: Must extract the final ridge step
            if hasattr(artifact, 'named_steps') and 'ridge' in artifact.named_steps:
                 model = artifact.named_steps['ridge']
            else:
                 # This error means the pipeline structure is not what's expected
                 return f"Error: Could not find 'ridge' step in the pipeline for {model_name}. Check the step name in your training code."
        elif 'Lasso' in model_name:
            # Pipeline model: Must extract the final ridge step
            if hasattr(artifact, 'named_steps') and 'lasso' in artifact.named_steps:
                 model = artifact.named_steps['lasso']
            else:
                 # This error means the pipeline structure is not what's expected
                 return f"Error: Could not find 'lasso' step in the pipeline for {model_name}. Check the step name in your training code."

        else:
            # Fallback for future models that might be stored directly
            model = artifact

        if model is None:
             return f"Error: Model extraction failed for {model_name}."
             
        # 3. Final Check and Return
        if not hasattr(model, 'coef_'):
            # This should now only trigger if the extracted object (e.g., model) 
            # is not a linear model but was extracted successfully.
            return f"Error: Extracted object type {type(model).__name__} does not have coefficients (coef_)."

        features_used = full_record['features_used']

        return {
            "coefficients": model.coef_.tolist(),
            "features": features_used 
        }
        
    except Exception as e:
        return f"Error loading or extracting coefficients: {str(e)}"


@mcp.tool()
def list_experiments() -> Dict[str, Any]:
    """
    Returns a summary of all registered experiment runs.
    """
    runs = experiment_registry.list_runs()
    
    # Format for client readability (showing only key info)
    summary_list = []
    for r in runs:
        summary_list.append({
            'run_id': r['run_id'],
            'timestamp': r['timestamp'],
            'model': r['model_config']['model_name'],
            'test_r2': r['metrics']['test_r2'],
            'test_rmse': r['metrics']['test_rmse'],
            'artifact': os.path.basename(r['artifact_path']),
        })
        
    return {"runs": summary_list}

# In MCP_utils.py

def apply_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full set of engineering transformations to the DataFrame.
    This logic MUST match the steps taken before training the production model.
    """
    
    # 1. Date Features (using year_sold, month_sold, day_of_week)
    data['year_sold'] = data.get('year_sold', 2014) # Default if missing
    
    # 2. Your custom engineered features
    
    # total_sqft and has_basement
    if 'sqft_above' in data.columns and 'sqft_basement' in data.columns:
        data['total_sqft'] = data['sqft_above'] + data['sqft_basement']
        data['has_basement'] = np.where(data['sqft_basement'] > 0, 1, 0)

    # living15_diff
    if {'sqft_living', 'sqft_living15'} <= set(data.columns):
        data['living15_diff'] = data['sqft_living'] - data['sqft_living15']

    # basement_share
    if {'sqft_basement', 'total_sqft'} <= set(data.columns):
        data['basement_share'] = np.where(
            data['total_sqft'] > 0,
            data['sqft_basement'] / data['total_sqft'],
            0
        )

    
    # Ratios (safe division)
    if 'sqft_living' in data.columns and 'sqft_lot' in data.columns:
        data['living_to_lot_ratio'] = np.where(data['sqft_lot'] > 0, data['sqft_living'] / data['sqft_lot'], 0)
        data['lot_per_living'] = np.where(data['sqft_living'] > 0, data['sqft_lot'] / data['sqft_living'], 0)
        
    if 'bathrooms' in data.columns and 'bedrooms' in data.columns:
        data['bath_per_bed'] = np.where(data['bedrooms'] > 0, data['bathrooms'] / data['bedrooms'], 0) 

    # Age and Renovation
    if 'year_sold' in data.columns and 'yr_built' in data.columns:
        data['house_age'] = data['year_sold'] - data['yr_built']
        
    if 'year_sold' in data.columns and 'yr_renovated' in data.columns:
        data['was_renovated'] = np.where(data['yr_renovated'] > 0, 1, 0)
        
        # Calculate years since last renovation, using built year if never renovated
        data['since_renovation'] = np.where(data['yr_renovated'] > 0, 
                                            data['year_sold'] - data['yr_renovated'], 
                                            data['year_sold'] - data['yr_built'])

    return data

# In MCP_utils.py

@mcp.tool()
def predict_house_price(run_id: str = None, raw_features_dict: Dict[str, Any] = None) -> Dict[str, Any] | str:
    """
    Loads a model artifact and generates a prediction. 
    Uses the current Production model if no run_id is provided.
    """
    
    # 1. Determine which Run ID to use (Default to Production)
    if run_id is None:
        run_id_to_use = production_registry.get_production_model()
        if run_id_to_use is None:
            return {"error": "No run_id provided, and no production model has been designated."}
    else:
        run_id_to_use = run_id
        
    # Input validation
    if raw_features_dict is None:
        return {"error": "Raw features must be provided."}
        
    # 2. Retrieve Metadata
    full_runs = experiment_registry.list_runs()
    full_record = next((r for r in full_runs if r['run_id'] == run_id_to_use), None)
    
    if not full_record:
        return {"error": f"Run ID '{run_id_to_use}' not found in the registry."}

    artifact_path = full_record.get('artifact_path')
    features_used = full_record['features_used']
    
    if not os.path.exists(artifact_path):
        return {"error": f"Model artifact not found at the recorded path: {artifact_path}"}
        
    try:
        # 3. Load Model Artifact
        with open(artifact_path, 'rb') as f:
            artifact = pickle.load(f)
            
        # 4. Feature Engineering and Preparation
        X_raw = pd.DataFrame([raw_features_dict]) 
        X_engineered = apply_feature_engineering(X_raw) 
        
        # Filter to the exact features used by the model
        X_final = X_engineered[features_used]
        
        # 5. Generate Prediction (The artifact handles scaling/transformations)
        prediction = artifact.predict(X_final)
        
        predicted_price = float(prediction[0])

        return {
            "status": "SUCCESS",
            "predicted_price": predicted_price,
            "model_run_id": run_id_to_use
        }
        
    except Exception as e:
        return {"error": f"Prediction failed due to server error: {str(e)}"}
    
# In MCP_utils.py

@mcp.tool()
def set_production_model() -> Dict[str, str] | str:
    """
    Designates the specified run_id as the current production model.
    """
    # 1. Validation Check: Ensure the run_id exists
    full_runs = experiment_registry.list_runs()
    best_run = max(
        full_runs,
        key=lambda x: x["metrics"]["test_r2"]  # or -test_rmse
    )
    # record_exists = any(r['run_id'] == run_id for r in full_runs)
    run_id = best_run['run_id']
    
    if not best_run:
        return f"Error: Run ID '{run_id}' not found in the Experiment Registry. Cannot promote."
        
    # 2. Set the status
    production_registry.set_production_model(run_id)
    
    return {"status": "SUCCESS", "message": f"Run ID {run_id} successfully promoted to Production."}


@mcp.tool()
def get_production_model() -> Dict[str, Any]:
    """
    Retrieves the currently designated production run_id and its summary metrics.
    """
    prod_id = production_registry.get_production_model()
    
    # Retrieve the summary of the production model for the client
    summary = experiment_registry.get_run_summary(prod_id) if prod_id else None
    
    return {
        "production_run_id": prod_id,
        "summary": summary
    }

@mcp.tool()
def get_model_list_summary() -> Dict[str, Any]:
    """
    Returns a summary of all registered experiments suitable for displaying 
    in a Streamlit app.
    """
    runs = experiment_registry.list_runs()
    
    # Format for client readability
    summary_list = []
    for r in runs:
        r2_val = r['metrics']['test_r2']
        rmse_val = r['metrics']['test_rmse']
        summary_list.append({
            'id': r['run_id'],
            'label': f"{r['model_config']['model_name']} ({r['model_config']['feature_set_name']} | R²: {r2_val:.4f})",
            'test_r2': r2_val,
            'test_rmse': rmse_val
        })
        
    return {"models": summary_list}

if __name__ == "__main__":
    mcp.run(transport='stdio')