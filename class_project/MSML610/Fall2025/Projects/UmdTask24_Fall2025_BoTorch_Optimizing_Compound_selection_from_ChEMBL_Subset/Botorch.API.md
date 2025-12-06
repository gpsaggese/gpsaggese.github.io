# BoTorch Multi-Objective Bayesian Optimization API

## Overview

This document describes the **native BoTorch API** for multi-objective Bayesian optimization and the **lightweight wrapper layer** (`Botorch_utils.py`) built on top of it for drug discovery applications.

**Use Case**: Multi-objective optimization for compound selection, balancing potency (pIC50) against cost metrics (molecular weight, synthetic accessibility).

---

## Table of Contents

1. [Native BoTorch API](#1-native-botorch-api)
   - [Core Components](#11-core-components)
   - [SingleTaskGP Model](#12-singletaskgp-model)
   - [Model Fitting](#13-model-fitting)
   - [Pareto Utilities](#14-pareto-utilities)
2. [Wrapper Layer API](#2-wrapper-layer-api)
   - [Data Structures](#21-data-structures)
   - [Protocols](#22-protocols)
   - [Molecular Utilities](#23-molecular-utilities)
   - [GP Model Utilities](#24-gp-model-utilities)
   - [Pareto Optimization](#25-pareto-optimization)
   - [Selection Strategies](#26-selection-strategies)
   - [Pipeline Functions](#27-pipeline-functions)
3. [Configuration](#3-configuration)
4. [Dependencies](#4-dependencies)

---

## 1. Native BoTorch API

### 1.1 Core Components

BoTorch provides building blocks for Bayesian optimization built on PyTorch and GPyTorch.

```python
# Core imports from BoTorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated

# GPyTorch for marginal log-likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
```

| Component | Module | Purpose |
|-----------|--------|---------|
| `SingleTaskGP` | `botorch.models` | Gaussian Process surrogate model |
| `fit_gpytorch_mll` | `botorch.fit` | Model fitting via MLL optimization |
| `is_non_dominated` | `botorch.utils.multi_objective.pareto` | Pareto dominance computation |
| `ExactMarginalLogLikelihood` | `gpytorch.mlls` | Loss function for GP training |

### 1.2 SingleTaskGP Model

The primary surrogate model for single-objective regression.

```python
from botorch.models import SingleTaskGP

class SingleTaskGP:
    """Gaussian Process model for single-task regression.
    
    Parameters
    ----------
    train_X : torch.Tensor
        Training features of shape (n_samples, n_features).
        Must be torch.float64.
    train_Y : torch.Tensor
        Training targets of shape (n_samples, 1).
        Must be torch.float64.
    likelihood : gpytorch.likelihoods.Likelihood, optional
        Gaussian likelihood (default: GaussianLikelihood).
    covar_module : gpytorch.kernels.Kernel, optional
        Covariance kernel (default: ScaleKernel(MaternKernel)).
    """
    
    def __init__(self, train_X, train_Y, likelihood=None, covar_module=None):
        ...
    
    def posterior(self, X, observation_noise=False):
        """Compute posterior distribution at X.
        
        Parameters
        ----------
        X : torch.Tensor
            Test points of shape (n_test, n_features).
        observation_noise : bool
            Whether to include observation noise.
            
        Returns
        -------
        GPyTorchPosterior
            Posterior with .mean and .variance attributes.
        """
        ...
    
    def forward(self, X):
        """Forward pass through the model."""
        ...
```

**Usage Example:**
```python
import torch
from botorch.models import SingleTaskGP

# Training data (must be float64)
X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
y_train = torch.tensor([[0.5], [1.5]], dtype=torch.float64)

# Create model
model = SingleTaskGP(X_train, y_train)

# Make predictions
model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test)
    mean = posterior.mean      # Shape: (n_test, 1)
    var = posterior.variance   # Shape: (n_test, 1)
```

### 1.3 Model Fitting

```python
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

def fit_gpytorch_mll(mll, optimizer=None, **kwargs):
    """Fit a GPyTorch model by maximizing marginal log-likelihood.
    
    Parameters
    ----------
    mll : gpytorch.mlls.MarginalLogLikelihood
        The marginal log-likelihood object.
    optimizer : callable, optional
        Optimizer (default: L-BFGS-B).
        
    Returns
    -------
    mll : MarginalLogLikelihood
        The fitted MLL object.
    """
    ...
```

**Usage Example:**
```python
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Create and fit model
model = SingleTaskGP(X_train, y_train)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)  # Optimizes hyperparameters in-place
```

### 1.4 Pareto Utilities

```python
from botorch.utils.multi_objective.pareto import is_non_dominated

def is_non_dominated(Y, deduplicate=True):
    """Identify Pareto-optimal (non-dominated) points.
    
    A point is non-dominated if no other point is better in ALL objectives.
    Assumes all objectives are to be MAXIMIZED.
    
    Parameters
    ----------
    Y : torch.Tensor
        Objective values of shape (n_points, n_objectives).
    deduplicate : bool
        Whether to consider duplicates as dominated.
        
    Returns
    -------
    torch.Tensor
        Boolean mask of shape (n_points,) where True indicates
        Pareto-optimal points.
    """
    ...
```

**Usage Example:**
```python
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

# Objectives: [potency, -cost] (both to maximize)
objectives = torch.tensor([
    [7.0, -0.3],  # Compound A
    [8.0, -0.5],  # Compound B
    [6.0, -0.2],  # Compound C
    [9.0, -0.8],  # Compound D
], dtype=torch.float64)

pareto_mask = is_non_dominated(objectives)
# Returns: tensor([False, True, True, True])
# B, C, D are Pareto-optimal; A is dominated by C
```

---

## 2. Wrapper Layer API

The wrapper layer (`Botorch_utils.py`) provides higher-level abstractions for drug discovery workflows.

### 2.1 Data Structures

#### OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    """Configuration for multi-objective Bayesian optimization.
    
    Attributes
    ----------
    feature_cols : List[str]
        Feature column names for GP models.
        Default: ['mol_weight', 'logp', 'hbd', 'hba', 
                  'rotatable_bonds', 'tpsa', 'sa_score', 'rings']
    test_size : float
        Fraction for test/candidate set (0.0 to 1.0). Default: 0.3
    random_state : int
        Random seed. Default: 42
    n_compounds_to_select : int
        Compounds per strategy. Default: 20
    mw_weight : float
        Molecular weight cost weight. Default: 0.5
    sa_weight : float
        Synthetic accessibility cost weight. Default: 0.5
    output_dir : str
        Results directory. Default: 'results'
    device : str
        PyTorch device. Default: 'cpu'
    """
```

#### MolecularDescriptors

```python
@dataclass
class MolecularDescriptors:
    """Container for molecular descriptors.
    
    Attributes
    ----------
    mol_weight : float    # Molecular weight (Da)
    logp : float          # Partition coefficient
    hbd : int             # H-bond donors
    hba : int             # H-bond acceptors
    rotatable_bonds : int # Rotatable bonds
    heavy_atoms : int     # Heavy atom count
    rings : int           # Ring count
    tpsa : float          # Topological polar surface area
    aromatic_rings : int  # Aromatic ring count
    amide_bonds : int     # Amide bond count
    num_heteroatoms : int # Heteroatom count
    sa_score : float      # Synthetic accessibility (1-10)
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...
```

#### GPModelMetrics

```python
@dataclass
class GPModelMetrics:
    """GP model evaluation metrics.
    
    Attributes
    ----------
    r2_score : float       # Coefficient of determination
    mse : float            # Mean squared error
    mae : float            # Mean absolute error
    objective_name : str   # Name of objective
    """
```

#### ParetoResult

```python
@dataclass
class ParetoResult:
    """Pareto front analysis results.
    
    Attributes
    ----------
    pareto_mask : torch.Tensor      # Boolean mask for Pareto points
    pareto_indices : torch.Tensor   # Indices of Pareto points
    pareto_compounds : pd.DataFrame # DataFrame of Pareto compounds
    n_pareto : int                  # Number of Pareto points
    percentage : float              # Percentage on Pareto front
    """
```

#### StrategyComparison

```python
@dataclass
class StrategyComparison:
    """Strategy comparison results.
    
    Attributes
    ----------
    strategy_name : str
    avg_potency : float
    avg_cost : float
    max_potency : float
    min_cost : float
    potency_cost_product : float
    selected_compounds : pd.DataFrame
    """
```

### 2.2 Protocols

Abstract interfaces defining contracts for extensibility.

```python
class MoleculeParser(Protocol):
    """Protocol for SMILES parsing."""
    def parse(self, smiles: str) -> Optional[Any]: ...
    def is_valid(self, smiles: str) -> bool: ...

class DescriptorCalculator(Protocol):
    """Protocol for descriptor calculation."""
    def calculate(self, mol: Any) -> MolecularDescriptors: ...
    def calculate_from_smiles(self, smiles: str) -> Optional[MolecularDescriptors]: ...

class SurrogateModel(Protocol):
    """Protocol for surrogate models."""
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None: ...
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def evaluate(self, X: torch.Tensor, y_true: np.ndarray) -> GPModelMetrics: ...

class ParetoOptimizer(Protocol):
    """Protocol for Pareto optimization."""
    def find_pareto_front(self, objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

class CompoundSelector(Protocol):
    """Protocol for compound selection."""
    def select(self, df: pd.DataFrame, n_select: int, strategy: SelectionStrategy) -> pd.DataFrame: ...
```

### 2.3 Molecular Utilities

```python
def safe_mol_from_smiles(smiles: str) -> Optional[Any]:
    """Convert SMILES to RDKit molecule safely.
    
    Returns None on failure instead of raising exceptions.
    """

def calculate_descriptors(mol: Any) -> Dict[str, Any]:
    """Calculate molecular descriptors for RDKit molecule.
    
    Returns empty dict on failure.
    """

def calculate_sa_score(mol: Any) -> float:
    """Calculate synthetic accessibility score (1-10 scale).
    
    Returns 10.0 (hardest) on failure.
    """

def calculate_cost_metric(
    mol_weight: float,
    sa_score: float,
    mw_weight: float = 0.5,
    sa_weight: float = 0.5,
    mw_max: float = 1000.0,
    sa_max: float = 10.0
) -> float:
    """Calculate normalized cost metric (0-1 scale)."""

def generate_molecular_descriptors(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    verbose: bool = True
) -> pd.DataFrame:
    """Generate all descriptors for DataFrame with SMILES column."""
```

### 2.4 GP Model Utilities

```python
def train_gp_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    objective_name: str = ""
) -> SingleTaskGP:
    """Train SingleTaskGP model.
    
    Parameters
    ----------
    X_train : torch.Tensor
        Training features (n_samples, n_features), dtype=float64.
    y_train : torch.Tensor
        Training targets (n_samples,), dtype=float64.
    objective_name : str
        Name for logging.
        
    Returns
    -------
    SingleTaskGP
        Fitted GP model.
    """

def predict_with_gp(
    gp_model: SingleTaskGP,
    X: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make predictions with trained GP.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (mean predictions, standard deviations)
    """

def evaluate_gp_model(
    gp_model: SingleTaskGP,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    objective_name: str = ""
) -> GPModelMetrics:
    """Evaluate GP model on test data.
    
    Returns GPModelMetrics with R², MSE, MAE.
    """
```

### 2.5 Pareto Optimization

```python
def find_pareto_front(
    objectives: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find Pareto-optimal points.
    
    Assumes all objectives are to be MAXIMIZED.
    
    Parameters
    ----------
    objectives : torch.Tensor
        Shape (n_points, n_objectives).
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (pareto_mask, pareto_indices)
    """

def compute_pareto_result(
    predicted_objectives: torch.Tensor,
    df_candidates: pd.DataFrame,
    mean_potency: torch.Tensor,
    mean_cost: torch.Tensor
) -> ParetoResult:
    """Compute complete Pareto analysis."""
```

### 2.6 Selection Strategies

```python
class SelectionStrategy(Enum):
    RANDOM = "random"
    TOP_POTENCY = "top_potency"
    LOWEST_COST = "lowest_cost"
    BALANCED_RATIO = "balanced_ratio"
    PARETO_OPTIMAL = "pareto_optimal"

def select_random(df: pd.DataFrame, n_select: int, random_state: int = 42) -> pd.DataFrame:
    """Random selection."""

def select_top_potency(df: pd.DataFrame, n_select: int, potency_col: str = "pIC50") -> pd.DataFrame:
    """Select highest potency compounds."""

def select_lowest_cost(df: pd.DataFrame, n_select: int, cost_col: str = "cost_metric") -> pd.DataFrame:
    """Select lowest cost compounds."""

def select_balanced_ratio(
    df: pd.DataFrame,
    n_select: int,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
    epsilon: float = 0.1
) -> pd.DataFrame:
    """Select best potency-to-cost ratio."""

def select_pareto_optimal(
    pareto_compounds: pd.DataFrame,
    n_select: int,
    sort_col: str = "predicted_potency"
) -> pd.DataFrame:
    """Select from Pareto front."""

def get_all_strategies(
    df_candidates: pd.DataFrame,
    pareto_compounds: pd.DataFrame,
    n_select: int = 20,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """Get selections from all strategies."""

def compare_strategies(
    strategies: Dict[str, pd.DataFrame],
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric"
) -> pd.DataFrame:
    """Compare strategy performance metrics."""
```

### 2.7 Pipeline Functions

```python
def prepare_optimization_data(
    df: pd.DataFrame,
    config: OptimizationConfig,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric"
) -> Dict[str, Any]:
    """Prepare data for optimization.
    
    Returns dict with:
    - X_train, X_test, y_train, y_test (numpy)
    - X_train_torch, X_test_torch, y_train_torch, y_test_torch
    - df_train, df_test
    - scaler_X
    """

def run_mobo_pipeline(
    df: pd.DataFrame,
    config: Optional[OptimizationConfig] = None,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
    verbose: bool = True
) -> Dict[str, Any]:
    """Run complete MOBO pipeline.
    
    Returns dict with:
    - data: Prepared data dict
    - gp_potency, gp_cost: Trained GP models
    - potency_metrics, cost_metrics: GPModelMetrics
    - pareto_result: ParetoResult
    - strategies: Dict of selected compounds per strategy
    - comparison_df: Strategy comparison DataFrame
    - config: OptimizationConfig used
    """

def export_results(
    results: Dict[str, Any],
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """Export results to CSV files.
    
    Exports:
    - pareto_optimal_compounds.csv
    - strategy_comparison.csv
    - selected_<strategy>.csv for each strategy
    
    Returns list of exported file paths.
    """
```

### 2.8 Visualization Functions

```python
def plot_pareto_front(
    df_candidates: pd.DataFrame,
    pareto_compounds: pd.DataFrame,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
    title: str = "Pareto Front: Potency vs Cost",
    save_path: Optional[str] = None
) -> None:
    """Plot Pareto front visualization."""

def plot_model_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model Validation",
    save_path: Optional[str] = None
) -> None:
    """Plot actual vs predicted values."""

def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "Potency_Cost_Product",
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None
) -> None:
    """Plot strategy comparison bar chart."""
```

---

## 3. Configuration

### Default Configuration

```python
from Botorch_utils import OptimizationConfig

config = OptimizationConfig(
    feature_cols=[
        'mol_weight', 'logp', 'hbd', 'hba',
        'rotatable_bonds', 'tpsa', 'sa_score', 'rings'
    ],
    test_size=0.3,
    random_state=42,
    n_compounds_to_select=20,
    mw_weight=0.5,
    sa_weight=0.5,
    output_dir='results',
    device='cpu'
)
```

### Custom Configuration

```python
config = OptimizationConfig(
    feature_cols=['mol_weight', 'logp', 'tpsa'],  # Subset of features
    test_size=0.2,                                 # 80/20 split
    n_compounds_to_select=50,                      # Select more compounds
    mw_weight=0.7,                                 # Prioritize MW in cost
    sa_weight=0.3,
    output_dir='my_results'
)
```

---

## 4. Dependencies

### Required Packages

```
torch>=2.0.0
botorch>=0.9.0
gpytorch>=1.10.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
rdkit>=2023.03.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Installation

```bash
pip install torch botorch gpytorch pandas numpy scikit-learn rdkit matplotlib seaborn
```

### Version Check

```python
import torch
import botorch
import gpytorch

print(f"PyTorch: {torch.__version__}")
print(f"BoTorch: {botorch.__version__}")
print(f"GPyTorch: {gpytorch.__version__}")
```

---

## Quick Reference

| Task | Native BoTorch | Wrapper Layer |
|------|----------------|---------------|
| Create GP model | `SingleTaskGP(X, y)` | `train_gp_model(X, y, name)` |
| Fit model | `fit_gpytorch_mll(mll)` | (included in train_gp_model) |
| Predict | `model.posterior(X).mean` | `predict_with_gp(model, X)` |
| Find Pareto | `is_non_dominated(Y)` | `find_pareto_front(Y)` |
| Full pipeline | Manual steps | `run_mobo_pipeline(df, config)` |
| Export results | Manual I/O | `export_results(results)` |
