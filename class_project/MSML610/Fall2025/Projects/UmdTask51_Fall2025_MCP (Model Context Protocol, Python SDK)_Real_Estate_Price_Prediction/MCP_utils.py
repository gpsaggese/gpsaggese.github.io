import mcp
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Ensure artifact directory exists
os.makedirs('artifacts', exist_ok=True)

def train_model(X_train, y_train, X_test, y_test, params: dict):
    """
    Trains an XGBoost model, tracking the run with MCP.
    """
    print(f"Starting MCP training run 'xgb-base-train' with params: {params}")
    
    with mcp.Context(name="xgb-base-train", params=params) as ctx:
        # Initialize and train the model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        
        # Log metrics and model artifact
        ctx.log_metrics({'test_mse': mse, 'test_rmse': np.sqrt(mse)})
        
        model_path = "artifacts/base_model.json"
        model.save_model(model_path)
        ctx.log_artifact(model_path, model)
        
        print(f"Training complete. Test MSE: {mse}")
        return model, {'test_mse': mse}

def tune_model(X_train, y_train, param_grid: dict):
    """
    Runs GridSearchCV for an XGBoost model, tracking with MCP.
    """
    print(f"Starting MCP tuning run 'xgb-tune'...")
    
    with mcp.Context(name="xgb-tune-run") as ctx:
        
        # Set up the model and grid search
        xgb = XGBRegressor(objective='reg:squarederror')
        grid_search = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid, 
            cv=3, 
            scoring='neg_mean_squared_error',
            verbose=1
        )
        
        # Run the search
        grid_search.fit(X_train, y_train)
        
        # Log the best results
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Invert score
        
        ctx.log_parameters(best_params)
        ctx.log_metrics({'best_cv_mse': best_score, 'best_cv_rmse': np.sqrt(best_score)})
        
        # Log the best model
        best_model = grid_search.best_estimator_
        model_path = "artifacts/tuned_model.json"
        best_model.save_model(model_path)
        ctx.log_artifact(model_path, best_model)
        
        print(f"Tuning complete. Best MSE: {best_score}")
        
        return best_model, best_params