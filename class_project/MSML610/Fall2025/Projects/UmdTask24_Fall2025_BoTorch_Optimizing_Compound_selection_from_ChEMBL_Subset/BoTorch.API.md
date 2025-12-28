# BoTorch API Documentation for Multi-Objective Drug Discovery

## Table of Contents
1. [Overview](#overview)
2. [Native BoTorch API](#native-botorch-api)
3. [Wrapper Layer Documentation](#wrapper-layer-documentation)
4. [Configuration Objects](#configuration-objects)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)

---

## Overview

This document describes the native BoTorch programming interface and the lightweight wrapper layer built on top of it for multi-objective Bayesian optimization in drug discovery applications.

**Purpose**: Optimize compound selection by simultaneously maximizing potency (pIC50) and minimizing synthetic cost/complexity.

**Key Technologies**:
- **BoTorch** v0.16.1 - Bayesian optimization framework
- **GPyTorch** v1.14.3 - Gaussian Process library
- **RDKit** 2025.9.3 - Cheminformatics toolkit
- **PyTorch** 2.9.0+ - Deep learning framework

---

## Native BoTorch API

### Core Classes

#### 1. `botorch.models.SingleTaskGP`
Gaussian Process model for single-task regression.

**Constructor**:
```python
from botorch.models import SingleTaskGP

model = SingleTaskGP(
    train_X: torch.Tensor,      # Shape: (n_samples, n_features)
    train_Y: torch.Tensor,      # Shape: (n_samples, 1)
    covar_module=None,          # Optional custom kernel
    mean_module=None,           # Optional custom mean function
    outcome_transform=None,     # Optional output transformation
    input_transform=None        # Optional input transformation
)
```

**Key Methods**:
- `posterior(X)`: Returns posterior distribution at test points
- `forward(X)`: Computes the GP prior distribution
- `eval()`: Sets model to evaluation mode
- `train()`: Sets model to training mode

**Parameters**:
- **train_X**: Training feature tensor (standardized)
- **train_Y**: Training target tensor (single column)
- Returns: Trained GP model object

---

#### 2. `botorch.fit.fit_gpytorch_mll`
Fits GP hyperparameters by maximizing marginal log-likelihood.

**Function Signature**:
```python
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(
    mll,                        # MarginalLogLikelihood object
    optimizer=fit_gpytorch_torch,
    options={'maxiter': 1000}   # Optional optimizer settings
)
```

**What It Does**:
- Optimizes kernel hyperparameters (lengthscales, noise variance)
- Uses L-BFGS-B optimization by default
- Returns fitted MLL object

**Typical Usage**:
```python
model = SingleTaskGP(X_train, y_train)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
# Model is now fitted and ready for predictions
```

---

#### 3. `botorch.utils.multi_objective.pareto.is_non_dominated`
Identifies Pareto-optimal points in multi-objective space.

**Function Signature**:
```python
from botorch.utils.multi_objective.pareto import is_non_dominated

pareto_mask = is_non_dominated(
    Y: torch.Tensor,           # Shape: (n_points, n_objectives)
    deduplicate: bool = True   # Remove duplicate points
)
```

**Returns**: Boolean tensor indicating which points are on the Pareto front.

**Algorithm**:
- A point is non-dominated if no other point is better in all objectives
- For maximization objectives, higher values are better
- Uses efficient vectorized comparison

**Example**:
```python
# Objectives: maximize potency, maximize -cost (i.e., minimize cost)
objectives = torch.tensor([
    [7.5, -0.3],  # Medium potency, low cost
    [8.5, -0.6],  # High potency, high cost
    [6.0, -0.2],  # Low potency, very low cost
    [7.0, -0.5],  # Dominated by point 0
])

pareto_mask = is_non_dominated(objectives)
# Result: [True, True, True, False]
# Points 0, 1, 2 are on Pareto front; point 3 is dominated
```

---

#### 4. `gpytorch.mlls.ExactMarginalLogLikelihood`
Computes the marginal log-likelihood for exact GP inference.

**Constructor**:
```python
from gpytorch.mlls import ExactMarginalLogLikelihood

mll = ExactMarginalLogLikelihood(
    likelihood,    # GaussianLikelihood object
    model          # SingleTaskGP model
)
```

**Purpose**:
- Provides objective function for hyperparameter optimization
- Computes \( \log p(y|X, \theta) \) where θ are hyperparameters

---

### Native API Workflow

```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood

# Step 1: Prepare data (standardize features)
X_train = torch.randn(100, 8, dtype=torch.float64)  # 100 compounds, 8 features
y_train = torch.randn(100, 1, dtype=torch.float64)  # Potency values

# Step 2: Build GP model
gp_model = SingleTaskGP(X_train, y_train)

# Step 3: Fit hyperparameters
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(mll)

# Step 4: Make predictions
X_test = torch.randn(50, 8, dtype=torch.float64)
gp_model.eval()
with torch.no_grad():
    posterior = gp_model.posterior(X_test)
    mean = posterior.mean        # Predicted values
    variance = posterior.variance  # Uncertainty estimates

# Step 5: Multi-objective optimization
# Assuming we have two objectives (potency, -cost)
y_potency = mean_potency_predictions
y_neg_cost = mean_neg_cost_predictions
objectives = torch.stack([y_potency, y_neg_cost], dim=-1)

pareto_mask = is_non_dominated(objectives)
pareto_indices = torch.where(pareto_mask)[0]
print(f"Found {len(pareto_indices)} Pareto-optimal compounds")
```

---

## Wrapper Layer Documentation

The wrapper layer (`botorch_utils.py`) provides high-level abstractions over the native BoTorch API to simplify common tasks in drug discovery applications.

### Design Principles

1. **Simplicity**: Hide complex BoTorch/GPyTorch details
2. **Reproducibility**: Built-in random seed management
3. **Validation**: Automatic data checks and error handling
4. **Extensibility**: Easy to add new objectives or constraints

### Architecture

```
┌─────────────────────────────────────────┐
│        User Application Code           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Wrapper Layer (botorch_utils.py)   │
│  • ChemDataProcessor                    │
│  • GPSurrogateModel                     │
│  • MultiObjectiveOptimizer              │
│  • StrategyComparator                   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Native BoTorch API              │
│  • SingleTaskGP                         │
│  • fit_gpytorch_mll                     │
│  • is_non_dominated                     │
└─────────────────────────────────────────┘
```

---

### Wrapper Classes

#### 1. `ChemDataProcessor`

**Purpose**: Handles molecular data processing, descriptor calculation, and cost metric computation.

**Constructor**:
```python
from botorch_utils import ChemDataProcessor

processor = ChemDataProcessor(
    cost_metric_weights: dict = {'mw': 0.5, 'sa': 0.5}
)
```

**Key Methods**:

##### `load_and_clean_data(filepath, config)`
Loads ChEMBL data and applies quality filters.

```python
df_cleaned = processor.load_and_clean_data(
    filepath='data.csv',
    config={
        'ic50_only': True,
        'remove_missing': True,
        'deduplicate': True
    }
)
```

**Returns**: Cleaned pandas DataFrame

**Data Quality Checks**:
- Filters for IC50 measurements in nM units
- Removes entries with missing SMILES or values
- Deduplicates by aggregating with median
- Validates SMILES strings using RDKit

---

##### `calculate_descriptors(df)`
Computes molecular descriptors for all compounds.

```python
df_with_descriptors = processor.calculate_descriptors(df)
```

**Generated Descriptors**:
- `mol_weight`: Molecular weight (Da)
- `logp`: Lipophilicity
- `hbd`: Hydrogen bond donors
- `hba`: Hydrogen bond acceptors
- `rotatable_bonds`: Rotatable bonds count
- `tpsa`: Topological polar surface area
- `rings`: Ring count
- `aromatic_rings`: Aromatic rings
- `heavy_atoms`: Heavy atom count
- `num_heteroatoms`: Heteroatom count
- `sa_score`: Synthetic accessibility (1-10 scale)

**Returns**: DataFrame with descriptor columns added

---

##### `compute_cost_metric(df, weights)`
Computes composite cost metric.

```python
df_with_cost = processor.compute_cost_metric(
    df,
    weights={'mw': 0.5, 'sa': 0.5}
)
```

**Formula**:
```
cost = w_mw * (MW / 1000) + w_sa * (SA_score / 10)
```

**Returns**: DataFrame with `cost_metric` column

---

#### 2. `GPSurrogateModel`

**Purpose**: Wrapper around BoTorch's SingleTaskGP with standardization and validation.

**Constructor**:
```python
from botorch_utils import GPSurrogateModel

gp_model = GPSurrogateModel(
    feature_cols: list,          # List of descriptor column names
    target_col: str,             # Name of target column (e.g., 'pIC50')
    standardize: bool = True,    # Standardize features
    device: str = 'cpu'          # 'cpu' or 'cuda'
)
```

**Key Methods**:

##### `fit(X_train, y_train)`
Trains the GP model and optimizes hyperparameters.

```python
gp_model.fit(X_train, y_train)
```

**Internal Steps**:
1. Standardizes features (if enabled)
2. Converts to torch tensors
3. Builds SingleTaskGP
4. Fits via MLL optimization
5. Stores scaler and model

**Returns**: Self (for method chaining)

---

##### `predict(X_test, return_std=False)`
Makes predictions on new data.

```python
# Point predictions only
predictions = gp_model.predict(X_test)

# With uncertainty
predictions, std_dev = gp_model.predict(X_test, return_std=True)
```

**Returns**:
- If `return_std=False`: numpy array of predictions
- If `return_std=True`: tuple (predictions, standard_deviations)

---

##### `validate(X_test, y_test)`
Computes validation metrics.

```python
metrics = gp_model.validate(X_test, y_test)
```

**Returns**: Dictionary with keys:
- `'r2'`: R² score
- `'rmse'`: Root mean squared error
- `'mae'`: Mean absolute error
- `'predictions'`: Array of predicted values

---

##### `get_model()`
Returns the underlying BoTorch model for advanced users.

```python
botorch_model = gp_model.get_model()
# Now you can use native BoTorch methods
posterior = botorch_model.posterior(X_torch)
```

---

#### 3. `MultiObjectiveOptimizer`

**Purpose**: Handles multi-objective Pareto front identification.

**Constructor**:
```python
from botorch_utils import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    objective_names: list,       # e.g., ['potency', 'cost']
    maximize: list               # e.g., [True, False]
)
```

**Key Methods**:

##### `fit_models(X_train, y_objectives)`
Trains separate GP models for each objective.

```python
optimizer.fit_models(
    X_train,           # Shape: (n_samples, n_features)
    y_objectives       # Shape: (n_samples, n_objectives)
)
```

**Returns**: Self

---

##### `predict_objectives(X_test)`
Predicts all objectives for test data.

```python
predictions = optimizer.predict_objectives(X_test)
# Shape: (n_test_samples, n_objectives)
```

---

##### `identify_pareto_front(X_candidates, y_predictions=None)`
Finds Pareto-optimal compounds.

```python
pareto_indices, pareto_objectives = optimizer.identify_pareto_front(
    X_candidates
)
```

**Returns**: Tuple of (indices, objective_values)

---

##### `rank_by_objective(pareto_indices, objective_idx, ascending=False)`
Ranks Pareto compounds by specific objective.

```python
# Rank by potency (descending)
ranked_indices = optimizer.rank_by_objective(
    pareto_indices,
    objective_idx=0,
    ascending=False
)
```

---

#### 4. `StrategyComparator`

**Purpose**: Compares multiple selection strategies.

**Constructor**:
```python
from botorch_utils import StrategyComparator

comparator = StrategyComparator(
    df_candidates: pd.DataFrame,
    objective_cols: list = ['pIC50', 'cost_metric']
)
```

**Key Methods**:

##### `add_strategy(name, selection_fn)`
Registers a selection strategy.

```python
def top_potency_strategy(df, n):
    return df.nlargest(n, 'pIC50')

comparator.add_strategy('Top Potency', top_potency_strategy)
```

---

##### `compare_all(n_select, metrics)`
Compares all registered strategies.

```python
results = comparator.compare_all(
    n_select=20,
    metrics=['avg_potency', 'avg_cost', 'max_potency', 'trade_off']
)
```

**Returns**: DataFrame with comparison metrics

---

##### `visualize_comparison(results, output_path)`
Creates comparison visualizations.

```python
comparator.visualize_comparison(
    results,
    output_path='comparison.png'
)
```

---

### Utility Functions

#### `safe_mol_from_smiles(smiles: str) -> Chem.Mol`
Safely converts SMILES to RDKit molecule object.

```python
from botorch_utils import safe_mol_from_smiles

mol = safe_mol_from_smiles('CCO')  # Ethanol
if mol is not None:
    # Process molecule
    mw = Descriptors.MolWt(mol)
```

---

#### `calculate_molecular_descriptors(mol: Chem.Mol) -> dict`
Computes comprehensive molecular descriptors.

```python
from botorch_utils import calculate_molecular_descriptors

descriptors = calculate_molecular_descriptors(mol)
# Returns: {'mol_weight': 46.07, 'logp': -0.31, ...}
```

---

#### `calculate_synthetic_accessibility(mol: Chem.Mol) -> float`
Estimates synthetic accessibility (1=easy, 10=hard).

```python
from botorch_utils import calculate_synthetic_accessibility

sa_score = calculate_synthetic_accessibility(mol)
# Range: 1.0 - 10.0
```

---

#### `visualize_pareto_front(df, x_col, y_col, pareto_mask, output_path)`
Creates Pareto front visualization.

```python
from botorch_utils import visualize_pareto_front

visualize_pareto_front(
    df=df_test,
    x_col='cost_metric',
    y_col='pIC50',
    pareto_mask=pareto_indices,
    output_path='pareto_front.png'
)
```

---

#### `export_compounds_for_ordering(df_selected, output_path)`
Exports compound list for experimental ordering.

```python
from botorch_utils import export_compounds_for_ordering

export_compounds_for_ordering(
    df_selected=top_compounds,
    output_path='compounds_to_order.csv'
)
```

**Output Columns**:
- `rank`: Selection rank
- `canonical_smiles`: SMILES string
- `inchi`: InChI identifier
- `pIC50`: Potency
- `cost_metric`: Cost score
- `mol_weight`: Molecular weight
- `sa_score`: Synthetic accessibility

---

## Configuration Objects

### Global Configuration

```python
CONFIG = {
    # Data paths
    'data_path': 'path/to/data.csv',
    'output_dir': 'results/',
    
    # Train-test split
    'test_size': 0.3,
    'random_state': 42,
    
    # Optimization settings
    'n_compounds_to_select': 20,
    
    # Computational
    'device': 'cpu',  # or 'cuda'
    'dtype': torch.float64,
    
    # Cost metric weights
    'cost_weights': {
        'mw': 0.5,      # Molecular weight component
        'sa': 0.5       # Synthetic accessibility component
    },
    
    # Feature columns
    'feature_cols': [
        'mol_weight', 'logp', 'hbd', 'hba',
        'rotatable_bonds', 'tpsa', 'rings', 'aromatic_rings'
    ],
    
    # GP hyperparameter bounds (optional)
    'gp_lengthscale_bounds': (0.1, 10.0),
    'gp_noise_bounds': (1e-6, 1e-2),
}
```

---

### Model Configuration

```python
GP_CONFIG = {
    'kernel': 'Matern',        # or 'RBF', 'RQ'
    'nu': 2.5,                 # Matern smoothness
    'ard': True,               # Automatic relevance determination
    'normalize_y': True,       # Standardize targets
    'optimizer': 'L-BFGS-B',
    'max_iter': 1000,
    'convergence_tol': 1e-6
}
```

---

### Optimization Configuration

```python
OPTIMIZATION_CONFIG = {
    'objectives': [
        {'name': 'potency', 'maximize': True},
        {'name': 'cost', 'maximize': False}
    ],
    
    # Pareto settings
    'deduplicate': True,
    'tolerance': 1e-6,
    
    # Strategy comparison
    'baseline_strategies': [
        'random',
        'top_potency',
        'lowest_cost',
        'balanced_ratio',
        'pareto_optimal'
    ]
}
```

---

## Usage Examples

### Example 1: Basic Workflow

```python
from botorch_utils import ChemDataProcessor, GPSurrogateModel, MultiObjectiveOptimizer

# 1. Load and process data
processor = ChemDataProcessor()
df = processor.load_and_clean_data('data.csv', CONFIG)
df = processor.calculate_descriptors(df)
df = processor.compute_cost_metric(df)

# 2. Train-test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# 3. Build GP models
X_train = train_df[CONFIG['feature_cols']].values
y_potency = train_df['pIC50'].values
y_cost = train_df['cost_metric'].values

gp_potency = GPSurrogateModel(CONFIG['feature_cols'], 'pIC50')
gp_potency.fit(X_train, y_potency)

gp_cost = GPSurrogateModel(CONFIG['feature_cols'], 'cost_metric')
gp_cost.fit(X_train, y_cost)

# 4. Multi-objective optimization
optimizer = MultiObjectiveOptimizer(
    objective_names=['potency', 'cost'],
    maximize=[True, False]
)
y_objectives = np.column_stack([y_potency, -y_cost])  # Negate cost for maximization
optimizer.fit_models(X_train, y_objectives)

# 5. Find Pareto front
X_test = test_df[CONFIG['feature_cols']].values
pareto_indices, pareto_objectives = optimizer.identify_pareto_front(X_test)

print(f"Found {len(pareto_indices)} Pareto-optimal compounds")
```

---

### Example 2: Custom Strategy

```python
from botorch_utils import StrategyComparator

comparator = StrategyComparator(test_df, ['pIC50', 'cost_metric'])

# Add custom strategy
def diversity_strategy(df, n):
    """Select diverse compounds using MaxMin algorithm."""
    # Implementation here
    return selected_df

comparator.add_strategy('Diversity MaxMin', diversity_strategy)

# Compare all strategies
results = comparator.compare_all(n_select=20)
comparator.visualize_comparison(results, 'strategy_comparison.png')
```

---

### Example 3: Active Learning Loop

```python
from botorch_utils import update_model_with_new_data

# Initial training
gp_model = GPSurrogateModel(feature_cols, 'pIC50')
gp_model.fit(X_train, y_train)

# Active learning iterations
for iteration in range(10):
    # 1. Predict on candidates
    predictions = gp_model.predict(X_candidates, return_std=True)
    
    # 2. Select next compound to test (highest uncertainty)
    next_idx = np.argmax(predictions[1])  # Max standard deviation
    
    # 3. Experimental testing (simulated here)
    true_value = conduct_experiment(X_candidates[next_idx])
    
    # 4. Update training data
    X_train = np.vstack([X_train, X_candidates[next_idx:next_idx+1]])
    y_train = np.append(y_train, true_value)
    
    # 5. Retrain model
    gp_model.fit(X_train, y_train)
    
    print(f"Iteration {iteration}: Added compound, R²={gp_model.validate(X_test, y_test)['r2']:.3f}")
```

---

## API Reference

### Quick Reference Table

| Component | Type | Purpose | Key Methods |
|-----------|------|---------|-------------|
| `ChemDataProcessor` | Class | Data processing | `load_and_clean_data()`, `calculate_descriptors()` |
| `GPSurrogateModel` | Class | GP modeling | `fit()`, `predict()`, `validate()` |
| `MultiObjectiveOptimizer` | Class | Pareto optimization | `fit_models()`, `identify_pareto_front()` |
| `StrategyComparator` | Class | Strategy comparison | `add_strategy()`, `compare_all()` |
| `safe_mol_from_smiles` | Function | SMILES parsing | Returns `Chem.Mol` or `None` |
| `calculate_molecular_descriptors` | Function | Descriptor computation | Returns `dict` of descriptors |
| `visualize_pareto_front` | Function | Visualization | Saves plot to file |

---

### Error Handling

All wrapper functions include error handling:

```python
try:
    result = gp_model.fit(X_train, y_train)
except ValueError as e:
    print(f"Data validation error: {e}")
except RuntimeError as e:
    print(f"Model fitting error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Common Errors**:
- `ValueError`: Invalid input shapes or types
- `RuntimeError`: GP optimization failed to converge
- `AttributeError`: Missing required columns in DataFrame

---

### Performance Considerations

**Computational Complexity**:
- GP training: O(n³) where n = number of training samples
- GP prediction: O(n²m) where m = number of test samples
- Pareto identification: O(n²k) where k = number of objectives

**Recommendations**:
- For n > 5000: Use sparse GP methods or subsample
- For multiple objectives (k > 3): Consider constraint handling
- Use GPU (`device='cuda'`) for faster matrix operations

---

### Version Compatibility

| Package | Minimum Version | Tested Version |
|---------|----------------|----------------|
| BoTorch | 0.8.0 | 0.16.1 |
| GPyTorch | 1.10 | 1.14.3 |
| PyTorch | 1.13 | 2.9.0 |
| RDKit | 2022.09 | 2025.9.3 |
| scikit-learn | 1.0 | 1.6.1 |
| pandas | 1.3 | 2.x |
| numpy | 1.21 | 2.0.2 |

---

## Best Practices

### 1. Data Preprocessing
```python
# Always standardize features for GP
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Model Validation
```python
# Always validate on held-out test set
metrics = gp_model.validate(X_test, y_test)
if metrics['r2'] < 0.5:
    print("Warning: Poor model performance")
```

### 3. Reproducibility
```python
# Set all random seeds
import random, numpy as np, torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 4. Cost Metric Design
```python
# Tune weights based on your priorities
cost_weights = {
    'mw': 0.3,   # Less weight on molecular weight
    'sa': 0.7    # More weight on synthetic accessibility
}
```

### 5. Pareto Front Validation
```python
# Check Pareto front size
if len(pareto_indices) < 5:
    print("Warning: Very few Pareto-optimal solutions")
elif len(pareto_indices) > 0.2 * len(candidates):
    print("Warning: Many Pareto solutions - objectives may be weakly correlated")
```

---

## Troubleshooting

### Issue 1: GP Optimization Not Converging
```python
# Increase max iterations
GP_CONFIG['max_iter'] = 2000

# Or add noise jitter
model.likelihood.noise = 1e-4
```

### Issue 2: Out of Memory
```python
# Reduce batch size or use CPU
CONFIG['device'] = 'cpu'

# Or subsample training data
X_train = X_train[:1000]
```

### Issue 3: Poor Predictions
```python
# Add more descriptors
additional_features = ['fraction_csp3', 'num_stereocenters']

# Or use fingerprints
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
```

---






