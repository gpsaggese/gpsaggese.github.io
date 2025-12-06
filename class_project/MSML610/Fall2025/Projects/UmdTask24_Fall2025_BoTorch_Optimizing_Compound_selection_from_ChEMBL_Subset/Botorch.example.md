# Multi-Objective Bayesian Optimization for Drug Discovery

A complete pipeline for identifying optimal drug candidates using **BoTorch** and **Gaussian Process** models, balancing potency maximization against synthesis cost minimization.

## Overview

This project implements a multi-objective optimization workflow for drug discovery, specifically targeting **Acetylcholinesterase (AChE)** inhibitors. The pipeline uses Bayesian optimization to identify Pareto-optimal compounds that achieve the best trade-offs between:

1. **Potency** (pIC50) — higher is better  
2. **Cost** (molecular weight + synthetic accessibility) — lower is better

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Data Loading & Cleaning | Load ChEMBL bioactivity data, filter for IC50 measurements, handle duplicates |
| 2. Molecular Descriptor Generation | Calculate descriptors using RDKit (MW, LogP, HBD, HBA, TPSA, etc.) |
| 3. Multi-Objective GP Modeling | Train separate Gaussian Process models for potency and cost prediction |
| 4. Pareto Front Optimization | Identify non-dominated compounds using BoTorch's `is_non_dominated` |
| 5. Strategy Comparison | Compare Pareto selection against baseline strategies |
| 6. Results Export | Save optimized compounds and visualizations |

## Requirements

```
torch>=2.0
botorch>=0.9
gpytorch
rdkit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Configuration

Key parameters in `CONFIG`:

```python
CONFIG = {
    'feature_cols': ['mol_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa', 'sa_score', 'rings'],
    'test_size': 0.3,
    'n_compounds_to_select': 20,
    'mw_weight': 0.5,  # Weight for molecular weight in cost metric
    'sa_weight': 0.5,  # Weight for synthetic accessibility in cost metric
}
```

## Molecular Descriptors

The pipeline calculates the following descriptors for each compound:

| Descriptor | Description |
|------------|-------------|
| `mol_weight` | Molecular weight |
| `logp` | Lipophilicity (partition coefficient) |
| `hbd` | Hydrogen bond donors |
| `hba` | Hydrogen bond acceptors |
| `rotatable_bonds` | Number of rotatable bonds |
| `tpsa` | Topological polar surface area |
| `sa_score` | Synthetic accessibility score (1=easy, 10=hard) |
| `rings` | Total ring count |

## GP Model Architecture

Two independent **SingleTaskGP** models are trained:

1. **Potency Model**: Predicts pIC50 from molecular descriptors  
2. **Cost Model**: Predicts negative cost metric (for maximization framing)

Both models use:
- Exact Gaussian Process inference
- Automatic hyperparameter optimization via `fit_gpytorch_mll`
- Standard scaling of input features

## Selection Strategies Compared

| Strategy | Description |
|----------|-------------|
| Random | Baseline random selection |
| Top Potency | Select highest pIC50 compounds |
| Lowest Cost | Select compounds with minimum cost metric |
| Balanced Ratio | Maximize pIC50/cost ratio |
| **Pareto-Optimal** | BoTorch multi-objective selection |

## Example Results

From the included analysis on AChE inhibitor data:

- **Total compounds analyzed**: 4,624  
- **Training set**: 3,236 compounds  
- **Candidate pool**: 1,388 compounds  
- **Pareto-optimal compounds**: 8 (0.6% of candidates)

**Model Performance**:
- Potency prediction R²: 0.526  
- Cost prediction R²: 1.000

**Pareto compounds vs. average**:
- +28.2% higher potency  
- -47.2% lower cost

## Output Files

```
results/
├── pareto_optimal_compounds.csv      # Pareto-optimal selections
├── strategy_comparison.csv           # Metrics for all strategies
├── selected_compounds_*.csv          # Compounds from each strategy
├── correlation_heatmap.png           # Descriptor correlations
└── *.png                             # Additional visualizations
```

## Usage

1. Set `raw_data_path` in CONFIG to your ChEMBL dataset  
2. Run cells sequentially through the notebook  
3. Results are exported to the `results/` directory

```python
# Quick start - load and process data
df = pd.read_csv(CONFIG['raw_data_path'])
df = clean_chembl_dataset(df)
df = generate_molecular_descriptors(df)
```

## Key Functions

```python
# Data cleaning
clean_chembl_dataset(df)              # Filter and validate bioactivity data
generate_molecular_descriptors(df)    # Calculate RDKit descriptors

# Modeling
train_gp_model(X_train, y_train)      # Train SingleTaskGP model

# Optimization
is_non_dominated(objectives)          # BoTorch Pareto front identification
```

## References

- [BoTorch Documentation](https://botorch.org/)
- [GPyTorch Documentation](https://gpytorch.ai/)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- ChEMBL Database for bioactivity data
