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
