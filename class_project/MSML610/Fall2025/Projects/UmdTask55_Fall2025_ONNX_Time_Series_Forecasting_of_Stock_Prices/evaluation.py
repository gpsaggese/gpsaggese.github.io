import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAPE value (percentage)
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        R² value
    """
    return float(r2_score(y_true, y_pred))


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Directional accuracy (percentage)
    """
    true_direction = np.diff(y_true.flatten()) > 0
    pred_direction = np.diff(y_pred.flatten()) > 0
    return float(np.mean(true_direction == pred_direction) * 100)


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with all metrics
    """
    return {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'Directional_Accuracy': calculate_directional_accuracy(y_true, y_pred)
    }


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = 'Predictions vs Actual',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Visualize predictions against actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Date index for x-axis
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if dates is not None:
        plt.plot(dates, y_true, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(y_true, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Time Steps')

    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_ensemble_predictions(
    *predictions: np.ndarray,
    weights: Optional[list] = None
) -> np.ndarray:
    """
    Create ensemble predictions by averaging multiple model predictions.

    Args:
        *predictions: Variable number of prediction arrays
        weights: Optional weights for weighted average (must sum to 1)

    Returns:
        Ensemble predictions
    """
    if len(predictions) == 0:
        raise ValueError("At least one prediction array must be provided")

    # Convert all to numpy arrays and ensure same shape
    pred_arrays = [np.asarray(p) for p in predictions]

    if weights is None:
        # Simple average
        ensemble = np.mean(pred_arrays, axis=0)
    else:
        # Weighted average
        if len(weights) != len(predictions):
            raise ValueError("Number of weights must match number of predictions")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        ensemble = np.zeros_like(pred_arrays[0])
        for pred, weight in zip(pred_arrays, weights):
            ensemble += weight * pred

    return ensemble


def compare_multiple_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stock_labels: Optional[np.ndarray] = None,
    stock_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare multiple models across potentially multiple stocks.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        stock_labels: Array indicating which stock each sample belongs to
        stock_names: List of unique stock names

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    if stock_labels is not None and stock_names is not None:
        # Per-stock comparison
        for stock in stock_names:
            stock_mask = stock_labels == stock

            for model_name, preds in predictions_dict.items():
                metrics = evaluate_forecasts(
                    y_true[stock_mask],
                    preds[stock_mask]
                )
                metrics['Stock'] = stock
                metrics['Model'] = model_name
                results.append(metrics)

        df = pd.DataFrame(results)
        # Reorder columns
        df = df[['Stock', 'Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]

    else:
        # Overall comparison
        for model_name, preds in predictions_dict.items():
            metrics = evaluate_forecasts(y_true, preds)
            metrics['Model'] = model_name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]
        df = df.sort_values('MAE')

    return df


def plot_ensemble_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stock_labels: Optional[np.ndarray] = None,
    stock_names: Optional[list] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12)
) -> None:
    """
    Plot ensemble comparison across multiple stocks.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        stock_labels: Array indicating which stock each sample belongs to
        stock_names: List of unique stock names
        dates: Date index for x-axis
        save_path: Path to save plot
        figsize: Figure size
    """
    if stock_labels is not None and stock_names is not None:
        # Create subplots for each stock
        n_stocks = len(stock_names)
        n_cols = 2
        n_rows = (n_stocks + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_stocks > 1 else [axes]

        for idx, stock in enumerate(stock_names):
            ax = axes[idx]
            stock_mask = stock_labels == stock

            y_stock = y_true[stock_mask]
            dates_stock = dates[stock_mask] if dates is not None else None

            if dates_stock is not None:
                ax.plot(dates_stock, y_stock, label='Actual', linewidth=2.5,
                       alpha=0.8, color='black')
                for model_name, preds in predictions_dict.items():
                    ax.plot(dates_stock, preds[stock_mask], label=model_name,
                           linewidth=2, alpha=0.7)
            else:
                ax.plot(y_stock, label='Actual', linewidth=2.5, alpha=0.8, color='black')
                for model_name, preds in predictions_dict.items():
                    ax.plot(preds[stock_mask], label=model_name, linewidth=2, alpha=0.7)

            ax.set_title(f'{stock} Predictions')
            ax.set_xlabel('Date' if dates_stock is not None else 'Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_stocks, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

    else:
        # Single plot for all data
        plt.figure(figsize=figsize)

        if dates is not None:
            plt.plot(dates, y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
            for model_name, preds in predictions_dict.items():
                plt.plot(dates, preds, label=model_name, linewidth=2, alpha=0.7)
            plt.xlabel('Date')
        else:
            plt.plot(y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
            for model_name, preds in predictions_dict.items():
                plt.plot(preds, label=model_name, linewidth=2, alpha=0.7)
            plt.xlabel('Time Steps')

        plt.ylabel('Value')
        plt.title('Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_performance_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = 'MAE',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create heatmap of model performance across stocks.

    Args:
        comparison_df: DataFrame from compare_multiple_models with Stock column
        metric: Metric to visualize
        save_path: Path to save plot
        figsize: Figure size
    """
    import seaborn as sns

    if 'Stock' not in comparison_df.columns:
        print("Warning: Stock column not found. Skipping heatmap.")
        return

    # Pivot data for heatmap
    pivot_data = comparison_df.pivot(index='Stock', columns='Model', values=metric)

    plt.figure(figsize=figsize)
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': metric})
    plt.title(f'Model Performance Heatmap ({metric})')
    plt.xlabel('Model')
    plt.ylabel('Stock')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def get_best_model_per_stock(
    comparison_df: pd.DataFrame,
    metric: str = 'MAE',
    lower_is_better: bool = True
) -> pd.DataFrame:
    """
    Identify best model for each stock based on metric.

    Args:
        comparison_df: DataFrame from compare_multiple_models with Stock column
        metric: Metric to use for comparison
        lower_is_better: Whether lower values are better

    Returns:
        DataFrame with best model per stock
    """
    if 'Stock' not in comparison_df.columns:
        # Overall best model
        if lower_is_better:
            best_idx = comparison_df[metric].idxmin()
        else:
            best_idx = comparison_df[metric].idxmax()

        return comparison_df.loc[[best_idx]]

    # Best model per stock
    best_models = []

    for stock in comparison_df['Stock'].unique():
        stock_data = comparison_df[comparison_df['Stock'] == stock]

        if lower_is_better:
            best_idx = stock_data[metric].idxmin()
        else:
            best_idx = stock_data[metric].idxmax()

        best_models.append(stock_data.loc[best_idx])

    return pd.DataFrame(best_models)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot residual analysis.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Path to save plot
        figsize: Figure size
    """
    residuals = y_true.flatten() - y_pred.flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(y_pred.flatten(), residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_ensemble_predictions(
    *predictions: np.ndarray,
    weights: Optional[list] = None
) -> np.ndarray:
    """
    Create ensemble predictions by averaging multiple model predictions.

    Args:
        *predictions: Variable number of prediction arrays
        weights: Optional weights for weighted average (must sum to 1)

    Returns:
        Ensemble predictions
    """
    if len(predictions) == 0:
        raise ValueError("At least one prediction array must be provided")

    # Convert all to numpy arrays and ensure same shape
    pred_arrays = [np.asarray(p) for p in predictions]

    if weights is None:
        # Simple average
        ensemble = np.mean(pred_arrays, axis=0)
    else:
        # Weighted average
        if len(weights) != len(predictions):
            raise ValueError("Number of weights must match number of predictions")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        ensemble = np.zeros_like(pred_arrays[0])
        for pred, weight in zip(pred_arrays, weights):
            ensemble += weight * pred

    return ensemble


def compare_multiple_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stock_labels: Optional[np.ndarray] = None,
    stock_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare multiple models across potentially multiple stocks.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        stock_labels: Array indicating which stock each sample belongs to
        stock_names: List of unique stock names

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    if stock_labels is not None and stock_names is not None:
        # Per-stock comparison
        for stock in stock_names:
            stock_mask = stock_labels == stock

            for model_name, preds in predictions_dict.items():
                metrics = evaluate_forecasts(
                    y_true[stock_mask],
                    preds[stock_mask]
                )
                metrics['Stock'] = stock
                metrics['Model'] = model_name
                results.append(metrics)

        df = pd.DataFrame(results)
        # Reorder columns
        df = df[['Stock', 'Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]

    else:
        # Overall comparison
        for model_name, preds in predictions_dict.items():
            metrics = evaluate_forecasts(y_true, preds)
            metrics['Model'] = model_name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]
        df = df.sort_values('MAE')

    return df


def plot_ensemble_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stock_labels: Optional[np.ndarray] = None,
    stock_names: Optional[list] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12)
) -> None:
    """
    Plot ensemble comparison across multiple stocks.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        stock_labels: Array indicating which stock each sample belongs to
        stock_names: List of unique stock names
        dates: Date index for x-axis
        save_path: Path to save plot
        figsize: Figure size
    """
    if stock_labels is not None and stock_names is not None:
        # Create subplots for each stock
        n_stocks = len(stock_names)
        n_cols = 2
        n_rows = (n_stocks + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_stocks > 1 else [axes]

        for idx, stock in enumerate(stock_names):
            ax = axes[idx]
            stock_mask = stock_labels == stock

            y_stock = y_true[stock_mask]
            dates_stock = dates[stock_mask] if dates is not None else None

            if dates_stock is not None:
                ax.plot(dates_stock, y_stock, label='Actual', linewidth=2.5,
                       alpha=0.8, color='black')
                for model_name, preds in predictions_dict.items():
                    ax.plot(dates_stock, preds[stock_mask], label=model_name,
                           linewidth=2, alpha=0.7)
            else:
                ax.plot(y_stock, label='Actual', linewidth=2.5, alpha=0.8, color='black')
                for model_name, preds in predictions_dict.items():
                    ax.plot(preds[stock_mask], label=model_name, linewidth=2, alpha=0.7)

            ax.set_title(f'{stock} Predictions')
            ax.set_xlabel('Date' if dates_stock is not None else 'Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_stocks, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

    else:
        # Single plot for all data
        plt.figure(figsize=figsize)

        if dates is not None:
            plt.plot(dates, y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
            for model_name, preds in predictions_dict.items():
                plt.plot(dates, preds, label=model_name, linewidth=2, alpha=0.7)
            plt.xlabel('Date')
        else:
            plt.plot(y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
            for model_name, preds in predictions_dict.items():
                plt.plot(preds, label=model_name, linewidth=2, alpha=0.7)
            plt.xlabel('Time Steps')

        plt.ylabel('Value')
        plt.title('Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_performance_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = 'MAE',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create heatmap of model performance across stocks.

    Args:
        comparison_df: DataFrame from compare_multiple_models with Stock column
        metric: Metric to visualize
        save_path: Path to save plot
        figsize: Figure size
    """
    import seaborn as sns

    if 'Stock' not in comparison_df.columns:
        print("Warning: Stock column not found. Skipping heatmap.")
        return

    # Pivot data for heatmap
    pivot_data = comparison_df.pivot(index='Stock', columns='Model', values=metric)

    plt.figure(figsize=figsize)
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': metric})
    plt.title(f'Model Performance Heatmap ({metric})')
    plt.xlabel('Model')
    plt.ylabel('Stock')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def get_best_model_per_stock(
    comparison_df: pd.DataFrame,
    metric: str = 'MAE',
    lower_is_better: bool = True
) -> pd.DataFrame:
    """
    Identify best model for each stock based on metric.

    Args:
        comparison_df: DataFrame from compare_multiple_models with Stock column
        metric: Metric to use for comparison
        lower_is_better: Whether lower values are better

    Returns:
        DataFrame with best model per stock
    """
    if 'Stock' not in comparison_df.columns:
        # Overall best model
        if lower_is_better:
            best_idx = comparison_df[metric].idxmin()
        else:
            best_idx = comparison_df[metric].idxmax()

        return comparison_df.loc[[best_idx]]

    # Best model per stock
    best_models = []

    for stock in comparison_df['Stock'].unique():
        stock_data = comparison_df[comparison_df['Stock'] == stock]

        if lower_is_better:
            best_idx = stock_data[metric].idxmin()
        else:
            best_idx = stock_data[metric].idxmax()

        best_models.append(stock_data.loc[best_idx])

    return pd.DataFrame(best_models)


def create_forecast_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model'
) -> pd.DataFrame:
    """
    Create a formatted report of forecast metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model

    Returns:
        DataFrame with metrics
    """
    metrics = evaluate_forecasts(y_true, y_pred)

    df = pd.DataFrame({
        'Model': [model_name],
        'MAE': [f"{metrics['MAE']:.4f}"],
        'RMSE': [f"{metrics['RMSE']:.4f}"],
        'MAPE': [f"{metrics['MAPE']:.2f}%"],
        'R²': [f"{metrics['R2']:.4f}"],
        'Direction Acc': [f"{metrics['Directional_Accuracy']:.2f}%"]
    })

    return df


def compare_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions

    Returns:
        DataFrame comparing all models
    """
    results = []

    for model_name, y_pred in predictions_dict.items():
        metrics = evaluate_forecasts(y_true, y_pred)
        metrics['Model'] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]

    return df.sort_values('MAE')


def plot_forecast_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = 'Model Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Plot multiple model predictions against actual values.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        dates: Date index for x-axis
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if dates is not None:
        plt.plot(dates, y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
        for model_name, y_pred in predictions_dict.items():
            plt.plot(dates, y_pred, label=model_name, linewidth=2, alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
        for model_name, y_pred in predictions_dict.items():
            plt.plot(y_pred, label=model_name, linewidth=2, alpha=0.7)
        plt.xlabel('Time Steps')

    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_ensemble_predictions(
    *predictions: np.ndarray,
    weights: Optional[list] = None
) -> np.ndarray:
    """
    Create ensemble predictions by averaging multiple model predictions.

    Args:
        *predictions: Variable number of prediction arrays
        weights: Optional weights for weighted average (must sum to 1)

    Returns:
        Ensemble predictions
    """
    if len(predictions) == 0:
        raise ValueError("At least one prediction array must be provided")

    # Convert all to numpy arrays and ensure same shape
    pred_arrays = [np.asarray(p) for p in predictions]

    if weights is None:
        # Simple average
        ensemble = np.mean(pred_arrays, axis=0)
    else:
        # Weighted average
        if len(weights) != len(predictions):
            raise ValueError("Number of weights must match number of predictions")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        ensemble = np.zeros_like(pred_arrays[0])
        for pred, weight in zip(pred_arrays, weights):
            ensemble += weight * pred

    return ensemble


def compare_multiple_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stock_labels: Optional[np.ndarray] = None,
    stock_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare multiple models across potentially multiple stocks.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        stock_labels: Array indicating which stock each sample belongs to
        stock_names: List of unique stock names

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    if stock_labels is not None and stock_names is not None:
        # Per-stock comparison
        for stock in stock_names:
            stock_mask = stock_labels == stock

            for model_name, preds in predictions_dict.items():
                metrics = evaluate_forecasts(
                    y_true[stock_mask],
                    preds[stock_mask]
                )
                metrics['Stock'] = stock
                metrics['Model'] = model_name
                results.append(metrics)

        df = pd.DataFrame(results)
        # Reorder columns
        df = df[['Stock', 'Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]

    else:
        # Overall comparison
        for model_name, preds in predictions_dict.items():
            metrics = evaluate_forecasts(y_true, preds)
            metrics['Model'] = model_name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']]
        df = df.sort_values('MAE')

    return df


def plot_ensemble_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stock_labels: Optional[np.ndarray] = None,
    stock_names: Optional[list] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12)
) -> None:
    """
    Plot ensemble comparison across multiple stocks.

    Args:
        y_true: Actual values
        predictions_dict: Dictionary mapping model names to predictions
        stock_labels: Array indicating which stock each sample belongs to
        stock_names: List of unique stock names
        dates: Date index for x-axis
        save_path: Path to save plot
        figsize: Figure size
    """
    if stock_labels is not None and stock_names is not None:
        # Create subplots for each stock
        n_stocks = len(stock_names)
        n_cols = 2
        n_rows = (n_stocks + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_stocks > 1 else [axes]

        for idx, stock in enumerate(stock_names):
            ax = axes[idx]
            stock_mask = stock_labels == stock

            y_stock = y_true[stock_mask]
            dates_stock = dates[stock_mask] if dates is not None else None

            if dates_stock is not None:
                ax.plot(dates_stock, y_stock, label='Actual', linewidth=2.5,
                       alpha=0.8, color='black')
                for model_name, preds in predictions_dict.items():
                    ax.plot(dates_stock, preds[stock_mask], label=model_name,
                           linewidth=2, alpha=0.7)
            else:
                ax.plot(y_stock, label='Actual', linewidth=2.5, alpha=0.8, color='black')
                for model_name, preds in predictions_dict.items():
                    ax.plot(preds[stock_mask], label=model_name, linewidth=2, alpha=0.7)

            ax.set_title(f'{stock} Predictions')
            ax.set_xlabel('Date' if dates_stock is not None else 'Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_stocks, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

    else:
        # Single plot for all data
        plt.figure(figsize=figsize)

        if dates is not None:
            plt.plot(dates, y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
            for model_name, preds in predictions_dict.items():
                plt.plot(dates, preds, label=model_name, linewidth=2, alpha=0.7)
            plt.xlabel('Date')
        else:
            plt.plot(y_true, label='Actual', linewidth=2.5, alpha=0.8, color='black')
            for model_name, preds in predictions_dict.items():
                plt.plot(preds, label=model_name, linewidth=2, alpha=0.7)
            plt.xlabel('Time Steps')

        plt.ylabel('Value')
        plt.title('Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_performance_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = 'MAE',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create heatmap of model performance across stocks.

    Args:
        comparison_df: DataFrame from compare_multiple_models with Stock column
        metric: Metric to visualize
        save_path: Path to save plot
        figsize: Figure size
    """
    import seaborn as sns

    if 'Stock' not in comparison_df.columns:
        print("Warning: Stock column not found. Skipping heatmap.")
        return

    # Pivot data for heatmap
    pivot_data = comparison_df.pivot(index='Stock', columns='Model', values=metric)

    plt.figure(figsize=figsize)
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': metric})
    plt.title(f'Model Performance Heatmap ({metric})')
    plt.xlabel('Model')
    plt.ylabel('Stock')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def get_best_model_per_stock(
    comparison_df: pd.DataFrame,
    metric: str = 'MAE',
    lower_is_better: bool = True
) -> pd.DataFrame:
    """
    Identify best model for each stock based on metric.

    Args:
        comparison_df: DataFrame from compare_multiple_models with Stock column
        metric: Metric to use for comparison
        lower_is_better: Whether lower values are better

    Returns:
        DataFrame with best model per stock
    """
    if 'Stock' not in comparison_df.columns:
        # Overall best model
        if lower_is_better:
            best_idx = comparison_df[metric].idxmin()
        else:
            best_idx = comparison_df[metric].idxmax()

        return comparison_df.loc[[best_idx]]

    # Best model per stock
    best_models = []

    for stock in comparison_df['Stock'].unique():
        stock_data = comparison_df[comparison_df['Stock'] == stock]

        if lower_is_better:
            best_idx = stock_data[metric].idxmin()
        else:
            best_idx = stock_data[metric].idxmax()

        best_models.append(stock_data.loc[best_idx])

    return pd.DataFrame(best_models)
