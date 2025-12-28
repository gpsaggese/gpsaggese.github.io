"""
Visualize Model Comparison Results

This script generates comprehensive visualizations from the model comparison results,
showcasing the performance of different regression models for house price prediction.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for professional-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_comparison_results(results_path):
    """Load model comparison results from JSON file."""
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    print(f"[OK] Loaded results for {len(results['comparison']['models'])} models\n")
    return results


def create_output_dir(output_dir):
    """Create output directory for visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {output_dir}\n")


def plot_cv_rmse_comparison(models_data, output_dir, best_model):
    """Plot cross-validation RMSE comparison across all models."""
    print("Creating CV RMSE comparison plot...")

    model_names = list(models_data.keys())
    cv_rmse = [models_data[name]['cv_mean_rmse'] for name in model_names]
    cv_std = [models_data[name]['cv_std_rmse'] for name in model_names]

    # Create color list - highlight best model
    colors = ['#2ecc71' if name == best_model else '#3498db' for name in model_names]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(model_names, cv_rmse, yerr=cv_std, capsize=5, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, rmse in zip(bars, cv_rmse):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation RMSE (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Cross-Validation RMSE\n(Lower is Better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Best Model'),
        Patch(facecolor='#3498db', label='Other Models')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: cv_rmse_comparison.png")


def plot_cv_score_distributions(models_data, output_dir, best_model):
    """Plot box plots showing CV score distributions."""
    print("Creating CV score distribution plot...")

    model_names = list(models_data.keys())
    cv_scores = [models_data[name]['cv_scores'] for name in model_names]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create box plot
    bp = ax.boxplot(cv_scores, labels=model_names, patch_artist=True, notch=True,
                    boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2))

    # Color boxes - highlight best model
    for patch, name in zip(bp['boxes'], model_names):
        if name == best_model:
            patch.set_facecolor('#2ecc71')
        else:
            patch.set_facecolor('#3498db')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation RMSE (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Score Distributions\n(5-Fold CV)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_score_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: cv_score_distributions.png")


def plot_training_time_comparison(models_data, output_dir):
    """Plot training time comparison."""
    print("Creating training time comparison plot...")

    model_names = list(models_data.keys())
    training_times = [models_data[name]['training_time'] for name in model_names]

    # Sort by training time
    sorted_indices = np.argsort(training_times)
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_times = [training_times[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(sorted_names, sorted_times, color='#9b59b6', alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, time in zip(bars, sorted_times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{time:.1f}s',
                ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Time Comparison\n(5-Fold Cross-Validation)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: training_time_comparison.png")


def plot_multi_metric_comparison(models_data, output_dir, best_model):
    """Plot comparison of multiple metrics (RMSE, MAE, R2)."""
    print("Creating multi-metric comparison plot...")

    model_names = list(models_data.keys())

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': model_names,
        'CV RMSE': [models_data[name]['cv_mean_rmse'] for name in model_names],
        'Train MAE': [models_data[name]['train_mae'] for name in model_names],
        'Train R²': [models_data[name]['train_r2'] for name in model_names]
    })

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: CV RMSE
    colors = ['#2ecc71' if name == best_model else '#3498db' for name in df['Model']]
    axes[0].bar(df['Model'], df['CV RMSE'], color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_title('Cross-Validation RMSE\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE (Log Scale)', fontsize=10, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Train MAE
    axes[1].bar(df['Model'], df['Train MAE'], color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].set_title('Training MAE\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('MAE (Log Scale)', fontsize=10, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    # Plot 3: Train R²
    axes[2].bar(df['Model'], df['Train R²'], color='#f39c12', alpha=0.8, edgecolor='black')
    axes[2].set_title('Training R² Score\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('R² Score', fontsize=10, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim([0.85, 1.0])

    plt.suptitle('Multi-Metric Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_metric_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: multi_metric_comparison.png")


def plot_cv_variability(models_data, output_dir):
    """Plot CV score variability (std deviation)."""
    print("Creating CV variability plot...")

    model_names = list(models_data.keys())
    cv_std = [models_data[name]['cv_std_rmse'] for name in model_names]
    cv_mean = [models_data[name]['cv_mean_rmse'] for name in model_names]

    # Calculate coefficient of variation (CV std / CV mean)
    cv_coefficient = [std / mean * 100 for std, mean in zip(cv_std, cv_mean)]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(model_names, cv_std, color='#e67e22', alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, std, coef in zip(bars, cv_std, cv_coefficient):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.4f}\n({coef:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('CV Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_title('Model Stability: Cross-Validation Variability\n(Lower is Better - More Stable)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_variability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: cv_variability.png")


def plot_performance_vs_time_tradeoff(models_data, output_dir, best_model):
    """Plot performance vs training time trade-off."""
    print("Creating performance-time trade-off plot...")

    model_names = list(models_data.keys())
    cv_rmse = [models_data[name]['cv_mean_rmse'] for name in model_names]
    training_times = [models_data[name]['training_time'] for name in model_names]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot scatter points
    for i, name in enumerate(model_names):
        if name == best_model:
            ax.scatter(training_times[i], cv_rmse[i], s=300, c='#2ecc71',
                      edgecolor='black', linewidth=2, zorder=5, label='Best Model')
        else:
            ax.scatter(training_times[i], cv_rmse[i], s=200, c='#3498db',
                      edgecolor='black', linewidth=1, alpha=0.7, zorder=3)

        # Add model name labels
        ax.annotate(name, (training_times[i], cv_rmse[i]),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Training Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation RMSE (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Training Time Trade-off\n(Bottom-Left is Ideal: Low RMSE, Fast Training)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_time_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: performance_time_tradeoff.png")


def create_summary_dashboard(models_data, output_dir, best_model, summary):
    """Create a comprehensive dashboard with key metrics."""
    print("Creating summary dashboard...")

    model_names = list(models_data.keys())

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. CV RMSE Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    cv_rmse = [models_data[name]['cv_mean_rmse'] for name in model_names]
    colors = ['#2ecc71' if name == best_model else '#3498db' for name in model_names]
    ax1.bar(model_names, cv_rmse, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Cross-Validation RMSE', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Best Model Info
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    best_info = f"""
    BEST MODEL

    {best_model}

    CV RMSE: {summary['best_rmse']:.4f}

    Improvement: {summary['rmse_improvement']:.4f}

    ({summary['rmse_improvement']/summary['worst_rmse']*100:.1f}% better than worst)
    """
    ax2.text(0.5, 0.5, best_info, transform=ax2.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3, pad=1))

    # 3. Training Time
    ax3 = fig.add_subplot(gs[1, 0])
    training_times = [models_data[name]['training_time'] for name in model_names]
    ax3.barh(model_names, training_times, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax3.set_title('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # 4. R² Scores
    ax4 = fig.add_subplot(gs[1, 1])
    r2_scores = [models_data[name]['train_r2'] for name in model_names]
    ax4.bar(model_names, r2_scores, color='#f39c12', alpha=0.8, edgecolor='black')
    ax4.set_title('Training R² Score', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim([0.85, 1.0])
    ax4.grid(axis='y', alpha=0.3)

    # 5. CV Variability
    ax5 = fig.add_subplot(gs[1, 2])
    cv_std = [models_data[name]['cv_std_rmse'] for name in model_names]
    ax5.bar(model_names, cv_std, color='#e67e22', alpha=0.8, edgecolor='black')
    ax5.set_title('CV Std Dev (Stability)', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Performance Rankings
    ax6 = fig.add_subplot(gs[2, :])
    rankings = list(reversed(models_data.keys()))[:5]  # Top 5
    rankings_rmse = [models_data[name]['cv_mean_rmse'] for name in rankings]
    y_pos = np.arange(len(rankings))
    colors_rank = ['#2ecc71' if name == best_model else '#3498db' for name in rankings]

    ax6.barh(y_pos, rankings_rmse, color=colors_rank, alpha=0.8, edgecolor='black')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f"#{i+1}: {name}" for i, name in enumerate(rankings)])
    ax6.invert_yaxis()
    ax6.set_xlabel('CV RMSE', fontsize=10, fontweight='bold')
    ax6.set_title('Top 5 Models by Performance', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # Add values
    for i, (name, rmse) in enumerate(zip(rankings, rankings_rmse)):
        ax6.text(rmse, i, f' {rmse:.4f}', va='center', fontweight='bold', fontsize=10)

    plt.suptitle('Model Comparison Dashboard - House Price Prediction',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: summary_dashboard.png")


def main():
    """Main function to generate all visualizations."""
    print("=" * 80)
    print("MODEL COMPARISON VISUALIZATION")
    print("=" * 80)
    print()

    # Configuration
    results_path = "models/comparison/comparison_results.json"
    output_dir = "docs/visualizations"

    # Load results
    results = load_comparison_results(results_path)
    models_data = results['comparison']['models']
    best_model = results['comparison']['best_model']
    summary = results['comparison']['summary']

    # Create output directory
    create_output_dir(output_dir)

    print("Generating visualizations...")
    print("-" * 60)

    # Generate all plots
    plot_cv_rmse_comparison(models_data, output_dir, best_model)
    plot_cv_score_distributions(models_data, output_dir, best_model)
    plot_training_time_comparison(models_data, output_dir)
    plot_multi_metric_comparison(models_data, output_dir, best_model)
    plot_cv_variability(models_data, output_dir)
    plot_performance_vs_time_tradeoff(models_data, output_dir, best_model)
    create_summary_dashboard(models_data, output_dir, best_model, summary)

    print("-" * 60)
    print(f"\n[OK] All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. cv_rmse_comparison.png - Main performance comparison")
    print("  2. cv_score_distributions.png - Score variability across folds")
    print("  3. training_time_comparison.png - Training efficiency")
    print("  4. multi_metric_comparison.png - RMSE, MAE, R² comparison")
    print("  5. cv_variability.png - Model stability")
    print("  6. performance_time_tradeoff.png - Performance vs efficiency")
    print("  7. summary_dashboard.png - Comprehensive overview")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
