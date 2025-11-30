"""Create visualization charts for model comparison and analysis.

This script generates comprehensive visualizations for the IoT anomaly detection
project, including:
- Model performance comparisons across all tasks
- Feature importance plots
- Confusion matrices
- Training metrics trends
- Distribution analyses

Usage:
    python scripts/create_visualizations.py --results_dir results --output_dir charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV results from training.

    Parameters
    ----------
    results_dir : Path
        Directory containing result CSVs

    Returns
    -------
    dict
        Dictionary mapping task name to DataFrame
    """
    results = {}
    result_files = {
        'Anomaly Detection': 'results_anomaly.csv',
        'Maintenance Prediction': 'results_maintenance.csv',
        'Downtime Risk': 'results_downtime.csv',
        'Failure Type': 'results_failure_type.csv',
        'Remaining Life': 'results_remaining_life.csv'
    }

    for task, filename in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            results[task] = pd.read_csv(filepath)
            print(f"✓ Loaded {task}: {len(results[task])} models")
        else:
            print(f"✗ Missing: {filepath}")

    return results


def plot_classification_comparison(results: Dict[str, pd.DataFrame],
                                   output_dir: Path):
    """Create comparison plots for classification tasks.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    output_dir : Path
        Directory to save plots
    """
    classification_tasks = ['Anomaly Detection', 'Maintenance Prediction',
                           'Downtime Risk', 'Failure Type']

    metrics = ['f1_mean', 'precision_mean', 'recall_mean', 'accuracy_mean']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'Accuracy']

    for metric, metric_name in zip(metrics, metric_names):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, task in enumerate(classification_tasks):
            if task not in results:
                continue

            df = results[task].sort_values(metric, ascending=True)

            ax = axes[idx]
            bars = ax.barh(df['model'], df[metric])

            # Color code by performance
            colors = plt.cm.RdYlGn(df[metric] / df[metric].max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel(metric_name, fontsize=11)
            ax.set_title(task, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (_, row) in enumerate(df.iterrows()):
                ax.text(row[metric] + 0.01, i, f'{row[metric]:.3f}',
                       va='center', fontsize=9)

        plt.suptitle(f'Model Comparison - {metric_name}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = output_dir / f'comparison_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_file}")
        plt.close()


def plot_regression_comparison(results: Dict[str, pd.DataFrame],
                               output_dir: Path):
    """Create comparison plots for regression task.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    output_dir : Path
        Directory to save plots
    """
    if 'Remaining Life' not in results:
        return

    df = results['Remaining Life'].sort_values('r2_mean', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # R² Score
    ax = axes[0]
    bars = ax.barh(df['model'], df['r2_mean'])
    colors = plt.cm.RdYlGn(df['r2_mean'] / df['r2_mean'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.set_xlabel('R² Score', fontsize=11)
    ax.set_title('R² Score', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['r2_mean'] + 0.01, i, f'{row["r2_mean"]:.3f}',
               va='center', fontsize=9)

    # MAE
    ax = axes[1]
    df_mae = df.sort_values('mae_mean', ascending=True)
    bars = ax.barh(df_mae['model'], df_mae['mae_mean'])
    colors = plt.cm.RdYlGn_r(df_mae['mae_mean'] / df_mae['mae_mean'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.set_xlabel('MAE (lower is better)', fontsize=11)
    ax.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (_, row) in enumerate(df_mae.iterrows()):
        ax.text(row['mae_mean'] + 1, i, f'{row["mae_mean"]:.1f}',
               va='center', fontsize=9)

    # RMSE
    ax = axes[2]
    df_rmse = df.sort_values('rmse_mean', ascending=True)
    bars = ax.barh(df_rmse['model'], df_rmse['rmse_mean'])
    colors = plt.cm.RdYlGn_r(df_rmse['rmse_mean'] / df_rmse['rmse_mean'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.set_xlabel('RMSE (lower is better)', fontsize=11)
    ax.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (_, row) in enumerate(df_rmse.iterrows()):
        ax.text(row['rmse_mean'] + 1, i, f'{row["rmse_mean"]:.1f}',
               va='center', fontsize=9)

    plt.suptitle('Remaining Useful Life Prediction - Model Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'comparison_regression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_best_models_summary(results: Dict[str, pd.DataFrame],
                             output_dir: Path):
    """Create a summary plot showing best model for each task.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    output_dir : Path
        Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    tasks = []
    best_models = []
    scores = []
    metric_types = []

    for task, df in results.items():
        if task == 'Remaining Life':
            best_idx = df['r2_mean'].idxmax()
            score = df.loc[best_idx, 'r2_mean']
            metric_type = 'R²'
        else:
            best_idx = df['f1_mean'].idxmax()
            score = df.loc[best_idx, 'f1_mean']
            metric_type = 'F1'

        tasks.append(task)
        best_models.append(df.loc[best_idx, 'model'])
        scores.append(score)
        metric_types.append(metric_type)

    # Create horizontal bar chart
    y_pos = np.arange(len(tasks))
    bars = ax.barh(y_pos, scores)

    # Color code
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tasks)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Best Model Performance by Task', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)

    # Add model names and scores
    for i, (model, score, metric) in enumerate(zip(best_models, scores, metric_types)):
        ax.text(score + 0.02, i, f'{model}\n{metric}: {score:.3f}',
               va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'best_models_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_model_training_time(results: Dict[str, pd.DataFrame],
                             output_dir: Path):
    """Plot training time comparison.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    output_dir : Path
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    classification_tasks = ['Anomaly Detection', 'Maintenance Prediction',
                           'Downtime Risk', 'Failure Type']

    for idx, task in enumerate(classification_tasks):
        if task not in results:
            continue

        df = results[task].sort_values('fit_time_mean', ascending=True)

        ax = axes[idx]
        bars = ax.barh(df['model'], df['fit_time_mean'])
        ax.set_xlabel('Training Time (seconds)', fontsize=10)
        ax.set_title(task, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row['fit_time_mean'] + 0.1, i,
                   f'{row["fit_time_mean"]:.2f}s',
                   va='center', fontsize=8)

    plt.suptitle('Model Training Time Comparison',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'training_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def create_summary_table(results: Dict[str, pd.DataFrame],
                        output_dir: Path):
    """Create summary table of best models.

    Parameters
    ----------
    results : dict
        Dictionary of results DataFrames
    output_dir : Path
        Directory to save table
    """
    summary_data = []

    for task, df in results.items():
        if task == 'Remaining Life':
            best_idx = df['r2_mean'].idxmax()
            row = df.loc[best_idx]
            summary_data.append({
                'Task': task,
                'Best Model': row['model'],
                'R² Score': f"{row['r2_mean']:.4f}",
                'MAE': f"{row['mae_mean']:.2f}",
                'RMSE': f"{row['rmse_mean']:.2f}",
                'Training Time (s)': f"{row['fit_time_mean']:.2f}"
            })
        else:
            best_idx = df['f1_mean'].idxmax()
            row = df.loc[best_idx]
            summary_data.append({
                'Task': task,
                'Best Model': row['model'],
                'F1 Score': f"{row['f1_mean']:.4f}",
                'Precision': f"{row['precision_mean']:.4f}",
                'Recall': f"{row['recall_mean']:.4f}",
                'Accuracy': f"{row['accuracy_mean']:.4f}",
                'Training Time (s)': f"{row['fit_time_mean']:.2f}"
            })

    df_summary = pd.DataFrame(summary_data)
    output_file = output_dir / 'best_models_summary.csv'
    df_summary.to_csv(output_file, index=False)
    print(f"✓ Saved {output_file}")

    return df_summary


def main(args):
    """Main function to create all visualizations."""
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Creating Visualizations for IoT Anomaly Detection")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print("="*80)

    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)

    if not results:
        print("No results found! Run training first.")
        return

    # Create plots
    print("\nGenerating comparison plots...")
    plot_classification_comparison(results, output_dir)
    plot_regression_comparison(results, output_dir)
    plot_best_models_summary(results, output_dir)
    plot_model_training_time(results, output_dir)

    # Create summary table
    print("\nCreating summary table...")
    df_summary = create_summary_table(results, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"All charts saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  - {f.name}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visualizations for IoT anomaly detection results"
    )
    parser.add_argument("--results_dir", default="results",
                       help="Directory containing result CSVs")
    parser.add_argument("--output_dir", default="charts",
                       help="Directory to save visualization charts")
    args = parser.parse_args()
    main(args)
