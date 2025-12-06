"""
Botorch_utils.py
================
Lightweight wrapper layer for BoTorch Multi-Objective Bayesian Optimization.

This module provides abstract interfaces (Protocols), dataclasses, and utility
functions for building multi-objective optimization pipelines using BoTorch.

The module is organized into:
1. Data Structures - Dataclasses for configuration and results
2. Protocols - Abstract interfaces for services
3. Molecular Utilities - RDKit-based descriptor calculation
4. GP Model Utilities - Gaussian Process training and prediction
5. Pareto Optimization - Multi-objective optimization utilities
6. Selection Strategies - Compound selection methods
7. Visualization - Plotting utilities
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 1: ENUMS AND CONSTANTS
# ============================================================================


class SelectionStrategy(Enum):
    """Available compound selection strategies."""

    RANDOM = "random"
    TOP_POTENCY = "top_potency"
    LOWEST_COST = "lowest_cost"
    BALANCED_RATIO = "balanced_ratio"
    PARETO_OPTIMAL = "pareto_optimal"


class ObjectiveDirection(Enum):
    """Direction of optimization for objectives."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


# ============================================================================
# SECTION 2: DATA STRUCTURES (DATACLASSES)
# ============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for multi-objective Bayesian optimization pipeline.

    Attributes:
        feature_cols: List of feature column names for GP models.
        test_size: Fraction of data for test/candidate set (0.0 to 1.0).
        random_state: Random seed for reproducibility.
        n_compounds_to_select: Number of compounds to select per strategy.
        mw_weight: Weight for molecular weight in cost metric.
        sa_weight: Weight for synthetic accessibility in cost metric.
        output_dir: Directory for saving results.
        device: PyTorch device ('cpu' or 'cuda').
    """

    feature_cols: List[str] = field(
        default_factory=lambda: [
            "mol_weight",
            "logp",
            "hbd",
            "hba",
            "rotatable_bonds",
            "tpsa",
            "sa_score",
            "rings",
        ]
    )
    test_size: float = 0.3
    random_state: int = 42
    n_compounds_to_select: int = 20
    mw_weight: float = 0.5
    sa_weight: float = 0.5
    output_dir: str = "results"
    device: str = "cpu"

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if self.n_compounds_to_select < 1:
            raise ValueError("n_compounds_to_select must be positive")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class MolecularDescriptors:
    """Container for molecular descriptors computed from SMILES.

    Attributes:
        mol_weight: Molecular weight (Da).
        logp: Partition coefficient (LogP).
        hbd: Number of hydrogen bond donors.
        hba: Number of hydrogen bond acceptors.
        rotatable_bonds: Number of rotatable bonds.
        heavy_atoms: Number of heavy atoms.
        rings: Total ring count.
        tpsa: Topological polar surface area.
        aromatic_rings: Number of aromatic rings.
        amide_bonds: Number of amide bonds.
        num_heteroatoms: Number of heteroatoms.
        sa_score: Synthetic accessibility score (1=easy, 10=hard).
    """

    mol_weight: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    heavy_atoms: int = 0
    rings: int = 0
    tpsa: float = 0.0
    aromatic_rings: int = 0
    amide_bonds: int = 0
    num_heteroatoms: int = 0
    sa_score: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptors to dictionary."""
        return {
            "mol_weight": self.mol_weight,
            "logp": self.logp,
            "hbd": self.hbd,
            "hba": self.hba,
            "rotatable_bonds": self.rotatable_bonds,
            "heavy_atoms": self.heavy_atoms,
            "rings": self.rings,
            "tpsa": self.tpsa,
            "aromatic_rings": self.aromatic_rings,
            "amide_bonds": self.amide_bonds,
            "num_heteroatoms": self.num_heteroatoms,
            "sa_score": self.sa_score,
        }


@dataclass
class GPModelMetrics:
    """Metrics for evaluating Gaussian Process model performance.

    Attributes:
        r2_score: Coefficient of determination (R²).
        mse: Mean squared error.
        mae: Mean absolute error.
        objective_name: Name of the objective being modeled.
    """

    r2_score: float
    mse: float
    mae: float
    objective_name: str = ""

    def __str__(self) -> str:
        return (
            f"{self.objective_name} Model:\n"
            f"  R² Score: {self.r2_score:.4f}\n"
            f"  MSE: {self.mse:.4f}\n"
            f"  MAE: {self.mae:.4f}"
        )


@dataclass
class ParetoResult:
    """Results from Pareto front optimization.

    Attributes:
        pareto_mask: Boolean tensor indicating Pareto-optimal points.
        pareto_indices: Indices of Pareto-optimal points.
        pareto_compounds: DataFrame of Pareto-optimal compounds.
        n_pareto: Number of Pareto-optimal points.
        percentage: Percentage of candidates on Pareto front.
    """

    pareto_mask: torch.Tensor
    pareto_indices: torch.Tensor
    pareto_compounds: pd.DataFrame
    n_pareto: int
    percentage: float


@dataclass
class StrategyComparison:
    """Comparison results for different selection strategies.

    Attributes:
        strategy_name: Name of the selection strategy.
        avg_potency: Average potency of selected compounds.
        avg_cost: Average cost of selected compounds.
        max_potency: Maximum potency achieved.
        min_cost: Minimum cost achieved.
        potency_cost_product: Trade-off score (potency / cost).
        selected_compounds: DataFrame of selected compounds.
    """

    strategy_name: str
    avg_potency: float
    avg_cost: float
    max_potency: float
    min_cost: float
    potency_cost_product: float
    selected_compounds: pd.DataFrame


# ============================================================================
# SECTION 3: PROTOCOLS (ABSTRACT INTERFACES)
# ============================================================================


class MoleculeParser(Protocol):
    """Protocol for parsing molecular structures from SMILES."""

    def parse(self, smiles: str) -> Optional[Any]:
        """Parse SMILES string to molecule object."""
        ...

    def is_valid(self, smiles: str) -> bool:
        """Check if SMILES string produces a valid molecule."""
        ...


class DescriptorCalculator(Protocol):
    """Protocol for calculating molecular descriptors."""

    def calculate(self, mol: Any) -> MolecularDescriptors:
        """Calculate descriptors for a molecule object."""
        ...

    def calculate_from_smiles(self, smiles: str) -> Optional[MolecularDescriptors]:
        """Calculate descriptors directly from SMILES string."""
        ...


class SurrogateModel(Protocol):
    """Protocol for surrogate models (e.g., Gaussian Processes)."""

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the model to training data."""
        ...

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance for new inputs."""
        ...

    def evaluate(
        self, X: torch.Tensor, y_true: np.ndarray
    ) -> GPModelMetrics:
        """Evaluate model performance on test data."""
        ...


class ParetoOptimizer(Protocol):
    """Protocol for multi-objective Pareto optimization."""

    def find_pareto_front(
        self, objectives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find Pareto-optimal points from objective values."""
        ...


class CompoundSelector(Protocol):
    """Protocol for compound selection strategies."""

    def select(
        self, df: pd.DataFrame, n_select: int, strategy: SelectionStrategy
    ) -> pd.DataFrame:
        """Select compounds according to a strategy."""
        ...


# ============================================================================
# SECTION 4: MOLECULAR UTILITIES
# ============================================================================


def safe_mol_from_smiles(smiles: str) -> Optional[Any]:
    """Safely convert SMILES to RDKit molecule.

    Args:
        smiles: SMILES string representation of molecule.

    Returns:
        RDKit molecule object or None if conversion fails.
    """
    from rdkit import Chem

    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def calculate_descriptors(mol: Any) -> Dict[str, Any]:
    """Calculate molecular descriptors for a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        Dictionary of descriptor name-value pairs.
    """
    from rdkit.Chem import Descriptors

    if mol is None:
        return {}
    try:
        return {
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "rings": Descriptors.RingCount(mol),
            "tpsa": Descriptors.TPSA(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol),
            "amide_bonds": Descriptors.NumAmideBonds(mol),
            "num_heteroatoms": Descriptors.NumHeteroatoms(mol),
            "num_aromatic_heterocycles": Descriptors.NumAromaticHeterocycles(mol),
            "num_aliphatic_heterocycles": Descriptors.NumAliphaticHeterocycles(mol),
        }
    except Exception:
        return {}


def calculate_sa_score(mol: Any) -> float:
    """Calculate synthetic accessibility score (heuristic).

    Args:
        mol: RDKit molecule object.

    Returns:
        SA score from 1 (easy) to 10 (hard).
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if mol is None:
        return 10.0
    try:
        mw = Descriptors.MolWt(mol)
        rings = Descriptors.RingCount(mol)
        try:
            chiral_centers = len(
                Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            )
        except Exception:
            chiral_centers = 0
        try:
            ring_info = mol.GetRingInfo()
            complex_rings = sum(1 for ring in ring_info.AtomRings() if len(ring) > 6)
        except Exception:
            complex_rings = 0

        sa_score = mw / 500 * 2 + rings * 0.5 + chiral_centers * 1.5 + complex_rings * 2
        return min(10, max(1, sa_score))
    except Exception:
        return 10.0


def calculate_cost_metric(
    mol_weight: float,
    sa_score: float,
    mw_weight: float = 0.5,
    sa_weight: float = 0.5,
    mw_max: float = 1000.0,
    sa_max: float = 10.0,
) -> float:
    """Calculate normalized cost metric combining MW and SA score.

    Args:
        mol_weight: Molecular weight.
        sa_score: Synthetic accessibility score.
        mw_weight: Weight for molecular weight component.
        sa_weight: Weight for SA score component.
        mw_max: Maximum MW for normalization.
        sa_max: Maximum SA score for normalization.

    Returns:
        Normalized cost metric (0-1 scale).
    """
    mw_norm = min(mol_weight / mw_max, 1.0)
    sa_norm = min(sa_score / sa_max, 1.0)
    return mw_weight * mw_norm + sa_weight * sa_norm


def generate_molecular_descriptors(
    df: pd.DataFrame, smiles_col: str = "canonical_smiles", verbose: bool = True
) -> pd.DataFrame:
    """Generate all molecular descriptors for a DataFrame.

    Args:
        df: DataFrame with SMILES column.
        smiles_col: Name of SMILES column.
        verbose: Whether to print progress.

    Returns:
        DataFrame with added descriptor columns.
    """
    if verbose:
        print("   Converting SMILES to molecules...")
    df = df.copy()
    df["molecule"] = df[smiles_col].apply(safe_mol_from_smiles)

    # Remove failed conversions
    initial_count = len(df)
    df = df[df["molecule"].notna()].copy()
    if verbose:
        print(f"   Valid molecules: {len(df)}/{initial_count}")

    # Calculate descriptors
    if verbose:
        print("   Calculating molecular descriptors...")
    descriptors_list = []
    for idx, mol in enumerate(df["molecule"]):
        if verbose and idx % 1000 == 0 and idx > 0:
            print(f"      Processed {idx}/{len(df)} molecules...")
        descriptors_list.append(calculate_descriptors(mol))

    descriptors_df = pd.DataFrame(descriptors_list)
    df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)

    # Calculate SA score
    if verbose:
        print("   Calculating synthetic accessibility scores...")
    df["sa_score"] = df["molecule"].apply(calculate_sa_score)

    # Drop molecule column
    df = df.drop(columns=["molecule"])

    if verbose:
        print(f"   Generated {len(descriptors_df.columns)} descriptors + SA score")
    return df


# ============================================================================
# SECTION 5: GP MODEL UTILITIES
# ============================================================================


def train_gp_model(
    X_train: torch.Tensor, y_train: torch.Tensor, objective_name: str = ""
) -> Any:
    """Train a SingleTaskGP model for a given objective.

    Args:
        X_train: Training features tensor.
        y_train: Training targets tensor (1D).
        objective_name: Name for logging.

    Returns:
        Trained SingleTaskGP model.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    print(f"Training GP for {objective_name}...")
    gp_model = SingleTaskGP(X_train, y_train.unsqueeze(-1))
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)
    print(f"✓ GP model trained for {objective_name}")
    return gp_model


def predict_with_gp(
    gp_model: Any, X: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make predictions with a trained GP model.

    Args:
        gp_model: Trained GP model.
        X: Input features tensor.

    Returns:
        Tuple of (mean predictions, standard deviations).
    """
    gp_model.eval()
    with torch.no_grad():
        posterior = gp_model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)
    return mean, std


def evaluate_gp_model(
    gp_model: Any,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    objective_name: str = "",
) -> GPModelMetrics:
    """Evaluate GP model performance on test data.

    Args:
        gp_model: Trained GP model.
        X_test: Test features tensor.
        y_test: True test values (numpy array).
        objective_name: Name for the metrics object.

    Returns:
        GPModelMetrics with R², MSE, and MAE.
    """
    mean, _ = predict_with_gp(gp_model, X_test)
    predictions = mean.numpy()

    return GPModelMetrics(
        r2_score=r2_score(y_test, predictions),
        mse=mean_squared_error(y_test, predictions),
        mae=mean_absolute_error(y_test, predictions),
        objective_name=objective_name,
    )


# ============================================================================
# SECTION 6: PARETO OPTIMIZATION UTILITIES
# ============================================================================


def find_pareto_front(objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find Pareto-optimal points from objective values.

    Assumes all objectives are to be MAXIMIZED.

    Args:
        objectives: Tensor of shape (n_points, n_objectives).

    Returns:
        Tuple of (pareto_mask, pareto_indices).
    """
    from botorch.utils.multi_objective.pareto import is_non_dominated

    pareto_mask = is_non_dominated(objectives)
    pareto_indices = torch.where(pareto_mask)[0]
    return pareto_mask, pareto_indices


def compute_pareto_result(
    predicted_objectives: torch.Tensor,
    df_candidates: pd.DataFrame,
    mean_potency: torch.Tensor,
    mean_cost: torch.Tensor,
) -> ParetoResult:
    """Compute complete Pareto analysis results.

    Args:
        predicted_objectives: Tensor of predicted objective values.
        df_candidates: DataFrame of candidate compounds.
        mean_potency: Mean potency predictions.
        mean_cost: Mean cost predictions.

    Returns:
        ParetoResult with all Pareto analysis data.
    """
    pareto_mask, pareto_indices = find_pareto_front(predicted_objectives)

    pareto_compounds = df_candidates.iloc[pareto_indices.numpy()].copy()
    pareto_compounds["predicted_potency"] = mean_potency[pareto_indices].numpy()
    pareto_compounds["predicted_neg_cost"] = mean_cost[pareto_indices].numpy()

    n_pareto = len(pareto_indices)
    percentage = 100 * n_pareto / len(df_candidates)

    return ParetoResult(
        pareto_mask=pareto_mask,
        pareto_indices=pareto_indices,
        pareto_compounds=pareto_compounds,
        n_pareto=n_pareto,
        percentage=percentage,
    )


# ============================================================================
# SECTION 7: SELECTION STRATEGIES
# ============================================================================


def select_random(
    df: pd.DataFrame, n_select: int, random_state: int = 42
) -> pd.DataFrame:
    """Select compounds randomly.

    Args:
        df: DataFrame of candidates.
        n_select: Number to select.
        random_state: Random seed.

    Returns:
        Selected compounds DataFrame.
    """
    return df.sample(n=min(n_select, len(df)), random_state=random_state)


def select_top_potency(
    df: pd.DataFrame, n_select: int, potency_col: str = "pIC50"
) -> pd.DataFrame:
    """Select compounds with highest potency.

    Args:
        df: DataFrame of candidates.
        n_select: Number to select.
        potency_col: Name of potency column.

    Returns:
        Selected compounds DataFrame.
    """
    return df.nlargest(n_select, potency_col)


def select_lowest_cost(
    df: pd.DataFrame, n_select: int, cost_col: str = "cost_metric"
) -> pd.DataFrame:
    """Select compounds with lowest cost.

    Args:
        df: DataFrame of candidates.
        n_select: Number to select.
        cost_col: Name of cost column.

    Returns:
        Selected compounds DataFrame.
    """
    return df.nsmallest(n_select, cost_col)


def select_balanced_ratio(
    df: pd.DataFrame,
    n_select: int,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
    epsilon: float = 0.1,
) -> pd.DataFrame:
    """Select compounds with best potency-to-cost ratio.

    Args:
        df: DataFrame of candidates.
        n_select: Number to select.
        potency_col: Name of potency column.
        cost_col: Name of cost column.
        epsilon: Small value to avoid division by zero.

    Returns:
        Selected compounds DataFrame.
    """
    df_copy = df.copy()
    df_copy["potency_cost_ratio"] = df_copy[potency_col] / (
        df_copy[cost_col] + epsilon
    )
    return df_copy.nlargest(n_select, "potency_cost_ratio")


def select_pareto_optimal(
    pareto_compounds: pd.DataFrame,
    n_select: int,
    sort_col: str = "predicted_potency",
) -> pd.DataFrame:
    """Select top compounds from Pareto front.

    Args:
        pareto_compounds: DataFrame of Pareto-optimal compounds.
        n_select: Number to select.
        sort_col: Column to sort by for selection.

    Returns:
        Selected compounds DataFrame.
    """
    return pareto_compounds.nlargest(min(n_select, len(pareto_compounds)), sort_col)


def get_all_strategies(
    df_candidates: pd.DataFrame,
    pareto_compounds: pd.DataFrame,
    n_select: int = 20,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Get selections from all available strategies.

    Args:
        df_candidates: DataFrame of candidate compounds.
        pareto_compounds: DataFrame of Pareto-optimal compounds.
        n_select: Number to select per strategy.
        random_state: Random seed for random selection.

    Returns:
        Dictionary mapping strategy names to selected DataFrames.
    """
    strategies = {
        "Random": select_random(df_candidates, n_select, random_state),
        "Top Potency": select_top_potency(df_candidates, n_select),
        "Lowest Cost": select_lowest_cost(df_candidates, n_select),
        "Balanced Ratio": select_balanced_ratio(df_candidates, n_select),
        "Pareto-Optimal (BoTorch)": select_pareto_optimal(pareto_compounds, n_select),
    }
    return strategies


def compare_strategies(
    strategies: Dict[str, pd.DataFrame],
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
) -> pd.DataFrame:
    """Compare performance of different selection strategies.

    Args:
        strategies: Dictionary of strategy name to selected compounds.
        potency_col: Name of potency column.
        cost_col: Name of cost column.

    Returns:
        DataFrame comparing strategy performance metrics.
    """
    results = []
    for name, selected in strategies.items():
        if len(selected) == 0:
            continue
        avg_potency = selected[potency_col].mean()
        avg_cost = selected[cost_col].mean()
        results.append(
            StrategyComparison(
                strategy_name=name,
                avg_potency=avg_potency,
                avg_cost=avg_cost,
                max_potency=selected[potency_col].max(),
                min_cost=selected[cost_col].min(),
                potency_cost_product=avg_potency / (avg_cost + 0.01),
                selected_compounds=selected,
            )
        )

    comparison_df = pd.DataFrame(
        [
            {
                "Strategy": r.strategy_name,
                "Avg_Potency": r.avg_potency,
                "Avg_Cost": r.avg_cost,
                "Max_Potency": r.max_potency,
                "Min_Cost": r.min_cost,
                "Potency_Cost_Product": r.potency_cost_product,
            }
            for r in results
        ]
    )
    return comparison_df.sort_values("Potency_Cost_Product", ascending=False)


# ============================================================================
# SECTION 8: DATA PREPARATION UTILITIES
# ============================================================================


def prepare_optimization_data(
    df: pd.DataFrame,
    config: OptimizationConfig,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
) -> Dict[str, Any]:
    """Prepare data for multi-objective optimization.

    Args:
        df: DataFrame with features and objectives.
        config: Optimization configuration.
        potency_col: Name of potency column.
        cost_col: Name of cost column.

    Returns:
        Dictionary with all prepared data tensors and scalers.
    """
    # Extract features and objectives
    X = df[config.feature_cols].values
    y_potency = df[potency_col].values
    y_cost = df[cost_col].values

    # Multi-objective: maximize potency, maximize -cost (minimize cost)
    y_objectives = np.column_stack([y_potency, -y_cost])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_objectives, test_size=config.test_size, random_state=config.random_state
    )

    df_train, df_test = train_test_split(
        df, test_size=config.test_size, random_state=config.random_state
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Convert to PyTorch tensors
    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float64)
    y_train_torch = torch.tensor(y_train, dtype=torch.float64)
    X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float64)
    y_test_torch = torch.tensor(y_test, dtype=torch.float64)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_torch": X_train_torch,
        "X_test_torch": X_test_torch,
        "y_train_torch": y_train_torch,
        "y_test_torch": y_test_torch,
        "df_train": df_train,
        "df_test": df_test,
        "scaler_X": scaler_X,
    }


# ============================================================================
# SECTION 9: VISUALIZATION UTILITIES
# ============================================================================


def plot_pareto_front(
    df_candidates: pd.DataFrame,
    pareto_compounds: pd.DataFrame,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
    title: str = "Pareto Front: Potency vs Cost",
    save_path: Optional[str] = None,
) -> None:
    """Plot Pareto front visualization.

    Args:
        df_candidates: DataFrame of all candidate compounds.
        pareto_compounds: DataFrame of Pareto-optimal compounds.
        potency_col: Name of potency column.
        cost_col: Name of cost column.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all candidates
    ax.scatter(
        df_candidates[cost_col],
        df_candidates[potency_col],
        alpha=0.3,
        s=30,
        c="gray",
        label="All Candidates",
    )

    # Plot Pareto front
    pareto_sorted = pareto_compounds.sort_values(cost_col)
    ax.scatter(
        pareto_sorted[cost_col],
        pareto_sorted[potency_col],
        s=150,
        c="red",
        marker="*",
        edgecolors="black",
        linewidths=1,
        label="Pareto-Optimal",
        zorder=5,
    )

    # Connect Pareto points
    ax.plot(
        pareto_sorted[cost_col],
        pareto_sorted[potency_col],
        "r--",
        alpha=0.7,
        linewidth=2,
    )

    ax.set_xlabel("Cost Metric (lower is better)", fontsize=12)
    ax.set_ylabel(f"{potency_col} (higher is better)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model Validation",
    save_path: Optional[str] = None,
) -> None:
    """Plot actual vs predicted values for model validation.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "Potency_Cost_Product",
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Plot bar chart comparing selection strategies.

    Args:
        comparison_df: DataFrame from compare_strategies().
        metric: Column name of metric to compare.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(comparison_df))
    bars = ax.bar(comparison_df["Strategy"], comparison_df[metric], color=colors)

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel(metric.replace("_", " "), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================================
# SECTION 10: HIGH-LEVEL PIPELINE FUNCTIONS
# ============================================================================


def run_mobo_pipeline(
    df: pd.DataFrame,
    config: Optional[OptimizationConfig] = None,
    potency_col: str = "pIC50",
    cost_col: str = "cost_metric",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run complete multi-objective Bayesian optimization pipeline.

    Args:
        df: DataFrame with molecular features and objectives.
        config: Optimization configuration (uses defaults if None).
        potency_col: Name of potency column.
        cost_col: Name of cost column.
        verbose: Whether to print progress.

    Returns:
        Dictionary with all pipeline results.
    """
    if config is None:
        config = OptimizationConfig()

    if verbose:
        print("=" * 60)
        print("MULTI-OBJECTIVE BAYESIAN OPTIMIZATION PIPELINE")
        print("=" * 60)

    # Step 1: Prepare data
    if verbose:
        print("\n[1/5] Preparing data...")
    data = prepare_optimization_data(df, config, potency_col, cost_col)

    # Step 2: Train GP models
    if verbose:
        print("\n[2/5] Training Gaussian Process models...")
    gp_potency = train_gp_model(
        data["X_train_torch"], data["y_train_torch"][:, 0], "Potency"
    )
    gp_cost = train_gp_model(
        data["X_train_torch"], data["y_train_torch"][:, 1], "Negative Cost"
    )

    # Step 3: Evaluate models
    if verbose:
        print("\n[3/5] Evaluating models...")
    potency_metrics = evaluate_gp_model(
        gp_potency, data["X_test_torch"], data["y_test"][:, 0], "Potency"
    )
    cost_metrics = evaluate_gp_model(
        gp_cost, data["X_test_torch"], data["y_test"][:, 1], "Negative Cost"
    )
    if verbose:
        print(potency_metrics)
        print(cost_metrics)

    # Step 4: Find Pareto front
    if verbose:
        print("\n[4/5] Computing Pareto front...")
    mean_potency, _ = predict_with_gp(gp_potency, data["X_test_torch"])
    mean_cost, _ = predict_with_gp(gp_cost, data["X_test_torch"])
    predicted_objectives = torch.stack([mean_potency, mean_cost], dim=-1)

    pareto_result = compute_pareto_result(
        predicted_objectives, data["df_test"], mean_potency, mean_cost
    )
    if verbose:
        print(f"   Found {pareto_result.n_pareto} Pareto-optimal compounds")
        print(f"   ({pareto_result.percentage:.1f}% of candidates)")

    # Step 5: Compare strategies
    if verbose:
        print("\n[5/5] Comparing selection strategies...")
    strategies = get_all_strategies(
        data["df_test"],
        pareto_result.pareto_compounds,
        config.n_compounds_to_select,
        config.random_state,
    )
    comparison_df = compare_strategies(strategies, potency_col, cost_col)

    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

    return {
        "data": data,
        "gp_potency": gp_potency,
        "gp_cost": gp_cost,
        "potency_metrics": potency_metrics,
        "cost_metrics": cost_metrics,
        "pareto_result": pareto_result,
        "strategies": strategies,
        "comparison_df": comparison_df,
        "config": config,
    }


# ============================================================================
# SECTION 11: EXPORT UTILITIES
# ============================================================================


def export_results(
    results: Dict[str, Any],
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """Export pipeline results to CSV files.

    Args:
        results: Dictionary from run_mobo_pipeline().
        output_dir: Output directory (uses config if None).
        verbose: Whether to print progress.

    Returns:
        List of exported file paths.
    """
    if output_dir is None:
        output_dir = results["config"].output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exported_files = []

    # Export Pareto-optimal compounds
    pareto_path = f"{output_dir}/pareto_optimal_compounds.csv"
    results["pareto_result"].pareto_compounds.to_csv(pareto_path, index=False)
    exported_files.append(pareto_path)
    if verbose:
        print(f"✓ Saved: {pareto_path}")

    # Export strategy comparison
    comparison_path = f"{output_dir}/strategy_comparison.csv"
    results["comparison_df"].to_csv(comparison_path, index=False)
    exported_files.append(comparison_path)
    if verbose:
        print(f"✓ Saved: {comparison_path}")

    # Export strategy selections
    for strategy_name, selected_df in results["strategies"].items():
        filename = f"selected_{strategy_name.replace(' ', '_').lower()}.csv"
        filepath = f"{output_dir}/{filename}"
        selected_df.to_csv(filepath, index=False)
        exported_files.append(filepath)
        if verbose:
            print(f"✓ Saved: {filepath}")

    return exported_files
