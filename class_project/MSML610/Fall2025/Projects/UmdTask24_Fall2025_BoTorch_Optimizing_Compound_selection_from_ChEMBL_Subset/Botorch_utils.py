"""
BoTorch Utilities Module

This module provides a lightweight wrapper layer on top of the native BoTorch API
for multi-objective Bayesian optimization in drug discovery applications.



Main Components:
- ChemDataProcessor: Molecular data processing and feature engineering
- GPSurrogateModel: Gaussian Process modeling wrapper
- MultiObjectiveOptimizer: Pareto optimization for multiple objectives
- StrategyComparator: Comparison of selection strategies
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Callable
from pathlib import Path

# BoTorch & GPyTorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# MOLECULAR DATA PROCESSING

class ChemDataProcessor:
    """
    Processes molecular data from ChEMBL-like datasets.
    
    Handles:
    - Data loading and cleaning
    - Molecular descriptor calculation
    - Cost metric computation
    - SMILES validation
    
    Example:
        >>> processor = ChemDataProcessor(cost_metric_weights={'mw': 0.5, 'sa': 0.5})
        >>> df = processor.load_and_clean_data('data.csv', config)
        >>> df = processor.calculate_descriptors(df)
        >>> df = processor.compute_cost_metric(df)
    """
    
    def __init__(self, cost_metric_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the data processor.
        
        Args:
            cost_metric_weights: Weights for cost metric components.
                                Keys: 'mw' (molecular weight), 'sa' (synthetic accessibility)
                                Default: {'mw': 0.5, 'sa': 0.5}
        """
        self.cost_weights = cost_metric_weights or {'mw': 0.5, 'sa': 0.5}
        self.descriptor_cols = []
    
    def load_and_clean_data(
        self,
        filepath: str,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load and clean ChEMBL-like bioactivity data.
        
        Args:
            filepath: Path to CSV file
            config: Optional configuration dict with keys:
                   - 'ic50_only': Filter for IC50 measurements (default: True)
                   - 'remove_missing': Remove entries with missing values (default: True)
                   - 'deduplicate': Aggregate duplicates by median (default: True)
        
        Returns:
            Cleaned pandas DataFrame
        
        Raises:
            FileNotFoundError: If filepath doesn't exist
            ValueError: If required columns are missing
        """
        config = config or {}
        ic50_only = config.get('ic50_only', True)
        remove_missing = config.get('remove_missing', True)
        deduplicate = config.get('deduplicate', True)
        
        # Load data
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        
        # Filter for IC50 measurements
        if ic50_only and 'standard_type' in df.columns:
            df = df[df['standard_type'] == 'IC50']
            print(f"After IC50 filter: {len(df)} records")
        
        # Remove missing values
        if remove_missing:
            required_cols = ['standard_value', 'canonical_smiles']
            existing_cols = [c for c in required_cols if c in df.columns]
            df = df.dropna(subset=existing_cols)
            print(f"After removing missing values: {len(df)} records")
        
        # Filter for exact measurements
        if 'standard_relation' in df.columns:
            df = df[df['standard_relation'] == '=']
            print(f"After exact measurements filter: {len(df)} records")
        
        # Convert IC50 to pIC50
        if 'standard_value' in df.columns:
            df['IC50_nM'] = pd.to_numeric(df['standard_value'], errors='coerce')
            df = df[df['IC50_nM'] > 0]
            df['pIC50'] = -np.log10(df['IC50_nM'] * 1e-9)
            print(f"Computed pIC50 for {len(df)} compounds")
        
        # Deduplicate by SMILES
        if deduplicate and 'canonical_smiles' in df.columns:
            original_len = len(df)
            df = df.groupby('canonical_smiles', as_index=False).agg({
                'IC50_nM': 'median',
                'pIC50': 'median'
            })
            print(f"After deduplication: {len(df)} unique compounds (removed {original_len - len(df)})")
        
        return df.reset_index(drop=True)
    
    def calculate_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate molecular descriptors for all compounds.
        
        Args:
            df: DataFrame with 'canonical_smiles' column
        
        Returns:
            DataFrame with descriptor columns added
        
        Descriptors calculated:
            - mol_weight: Molecular weight (Da)
            - logp: Lipophilicity
            - hbd: Hydrogen bond donors
            - hba: Hydrogen bond acceptors
            - rotatable_bonds: Number of rotatable bonds
            - tpsa: Topological polar surface area
            - rings: Ring count
            - aromatic_rings: Aromatic ring count
            - heavy_atoms: Heavy atom count
            - num_heteroatoms: Heteroatom count
            - sa_score: Synthetic accessibility (1=easy, 10=hard)
        """
        if 'canonical_smiles' not in df.columns:
            raise ValueError("DataFrame must have 'canonical_smiles' column")
        
        print(f"Calculating descriptors for {len(df)} compounds...")
        
        descriptor_rows = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            mol = safe_mol_from_smiles(row['canonical_smiles'])
            if mol is not None:
                desc = calculate_molecular_descriptors(mol)
                desc['sa_score'] = calculate_synthetic_accessibility(mol)
                descriptor_rows.append(desc)
                valid_indices.append(idx)
        
        # Create descriptor DataFrame
        desc_df = pd.DataFrame(descriptor_rows, index=valid_indices)
        
        # Merge with original DataFrame
        result_df = df.loc[valid_indices].copy()
        result_df = pd.concat([result_df, desc_df], axis=1)
        
        self.descriptor_cols = list(desc_df.columns)
        print(f"Calculated {len(self.descriptor_cols)} descriptors for {len(result_df)} compounds")
        
        return result_df.reset_index(drop=True)
    
    def compute_cost_metric(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute composite cost metric.
        
        Args:
            df: DataFrame with 'mol_weight' and 'sa_score' columns
            weights: Optional custom weights (default: use instance weights)
        
        Returns:
            DataFrame with 'cost_metric' column added
        
        Formula:
            cost = w_mw * (MW / 1000) + w_sa * (SA_score / 10)
        """
        weights = weights or self.cost_weights
        
        required_cols = ['mol_weight', 'sa_score']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df.copy()
        
        # Normalize components
        mw_norm = df['mol_weight'] / 1000.0
        sa_norm = df['sa_score'] / 10.0
        
        # Weighted combination
        df['cost_metric'] = (
            weights.get('mw', 0.5) * mw_norm +
            weights.get('sa', 0.5) * sa_norm
        )
        
        print(f"Computed cost metric (MW weight={weights.get('mw', 0.5)}, SA weight={weights.get('sa', 0.5)})")
        print(f"Cost range: [{df['cost_metric'].min():.3f}, {df['cost_metric'].max():.3f}]")
        
        return df


# GAUSSIAN PROCESS SURROGATE MODEL


class GPSurrogateModel:
    """
    Wrapper around BoTorch's SingleTaskGP for easier usage.
    
    Handles:
    - Feature standardization
    - Model training and hyperparameter optimization
    - Prediction with uncertainty
    - Validation metrics
    
    Example:
        >>> model = GPSurrogateModel(feature_cols=['mol_weight', 'logp'], target_col='pIC50')
        >>> model.fit(X_train, y_train)
        >>> predictions, std = model.predict(X_test, return_std=True)
        >>> metrics = model.validate(X_test, y_test)
    """
    
    def __init__(
        self,
        feature_cols: List[str],
        target_col: str,
        standardize: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize GP surrogate model.
        
        Args:
            feature_cols: List of feature column names
            target_col: Name of target column
            standardize: Whether to standardize features (recommended)
            device: 'cpu' or 'cuda'
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.standardize = standardize
        self.device = device
        
        self.scaler = StandardScaler() if standardize else None
        self.model = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPSurrogateModel':
        """
        Train the GP model.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)
        
        Returns:
            Self (for method chaining)
        
        Raises:
            ValueError: If input shapes are invalid
        """
        if X.shape[0] != len(y):
            raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, y has {len(y)}")
        
        if X.shape[1] != len(self.feature_cols):
            raise ValueError(f"X has {X.shape[1]} features but {len(self.feature_cols)} expected")
        
        print(f"\nTraining GP model for {self.target_col}...")
        print(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # Standardize features
        if self.standardize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Convert to torch tensors
        X_torch = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        y_torch = torch.tensor(y.reshape(-1, 1), dtype=torch.float64, device=self.device)
        
        # Build GP model
        self.model = SingleTaskGP(X_torch, y_torch)
        
        # Fit hyperparameters via MLL
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        
        self.is_fitted = True
        print(f"✓ GP model trained successfully")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            return_std: If True, also return standard deviations
        
        Returns:
            If return_std=False: predictions (n_samples,)
            If return_std=True: (predictions, std_devs)
        
        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Standardize features
        if self.standardize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Convert to torch tensor
        X_torch = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_torch)
            mean = posterior.mean.cpu().numpy().flatten()
            
            if return_std:
                std = posterior.variance.sqrt().cpu().numpy().flatten()
                return mean, std
            else:
                return mean
    
    def validate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Compute validation metrics.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
        
        Returns:
            Dictionary with keys: 'r2', 'rmse', 'mae', 'predictions'
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'predictions': predictions
        }
        
        return metrics
    
    def get_model(self) -> SingleTaskGP:
        """
        Get the underlying BoTorch model for advanced users.
        
        Returns:
            BoTorch SingleTaskGP model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model



# MULTI-OBJECTIVE OPTIMIZATION


class MultiObjectiveOptimizer:
    """
    Handles multi-objective Pareto front identification.
    
    Trains separate GP models for each objective and identifies
    Pareto-optimal solutions.
    
    Example:
        >>> optimizer = MultiObjectiveOptimizer(
        ...     objective_names=['potency', 'cost'],
        ...     maximize=[True, False]
        ... )
        >>> optimizer.fit_models(X_train, y_objectives)
        >>> pareto_idx, pareto_obj = optimizer.identify_pareto_front(X_test)
    """
    
    def __init__(
        self,
        objective_names: List[str],
        maximize: List[bool],
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objective_names: Names of objectives (e.g., ['potency', 'cost'])
            maximize: List of bools indicating whether to maximize each objective
            feature_cols: Optional list of feature column names
        """
        self.objective_names = objective_names
        self.maximize = maximize
        self.feature_cols = feature_cols or []
        self.models = {}
        self.is_fitted = False
    
    def fit_models(
        self,
        X: np.ndarray,
        y_objectives: np.ndarray
    ) -> 'MultiObjectiveOptimizer':
        """
        Train GP models for each objective.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y_objectives: Objective matrix, shape (n_samples, n_objectives)
        
        Returns:
            Self (for method chaining)
        """
        if y_objectives.shape[1] != len(self.objective_names):
            raise ValueError(
                f"y_objectives has {y_objectives.shape[1]} columns but "
                f"{len(self.objective_names)} objectives expected"
            )
        
        print(f"\nTraining {len(self.objective_names)} GP models...")
        
        for i, (name, maximize_obj) in enumerate(zip(self.objective_names, self.maximize)):
            # For BoTorch, we always maximize, so negate if needed
            y_obj = y_objectives[:, i] if maximize_obj else -y_objectives[:, i]
            
            model = GPSurrogateModel(
                feature_cols=self.feature_cols or [f'f{j}' for j in range(X.shape[1])],
                target_col=name,
                standardize=True
            )
            model.fit(X, y_obj)
            self.models[name] = model
        
        self.is_fitted = True
        print("✓ All objective models trained")
        
        return self
    
    def predict_objectives(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict all objectives for test data.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            return_std: If True, also return standard deviations
        
        Returns:
            If return_std=False: predictions, shape (n_samples, n_objectives)
            If return_std=True: (predictions, std_devs)
        """
        if not self.is_fitted:
            raise RuntimeError("Models must be fitted before prediction")
        
        predictions = []
        stds = [] if return_std else None
        
        for i, (name, maximize_obj) in enumerate(zip(self.objective_names, self.maximize)):
            if return_std:
                pred, std = self.models[name].predict(X, return_std=True)
                # Undo negation if we minimized
                if not maximize_obj:
                    pred = -pred
                predictions.append(pred)
                stds.append(std)
            else:
                pred = self.models[name].predict(X)
                if not maximize_obj:
                    pred = -pred
                predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        if return_std:
            stds = np.column_stack(stds)
            return predictions, stds
        else:
            return predictions
    
    def identify_pareto_front(
        self,
        X: np.ndarray,
        y_predictions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify Pareto-optimal solutions.
        
        Args:
            X: Feature matrix for candidates
            y_predictions: Optional pre-computed predictions
                          If None, will predict using fitted models
        
        Returns:
            Tuple of (pareto_indices, pareto_objectives)
        """
        if y_predictions is None:
            y_predictions = self.predict_objectives(X)
        
        # Convert to torch for is_non_dominated
        y_torch = torch.tensor(y_predictions, dtype=torch.float64)
        
        # For BoTorch, all objectives should be maximization
        # We already handled negation in predict_objectives
        # So convert minimize objectives back to maximize
        y_for_pareto = y_torch.clone()
        for i, maximize_obj in enumerate(self.maximize):
            if not maximize_obj:
                y_for_pareto[:, i] = -y_for_pareto[:, i]
        
        # Identify Pareto front
        pareto_mask = is_non_dominated(y_for_pareto)
        pareto_indices = torch.where(pareto_mask)[0].numpy()
        pareto_objectives = y_predictions[pareto_indices]
        
        print(f"\nPareto Front Analysis:")
        print(f"Total candidates: {len(X)}")
        print(f"Pareto-optimal: {len(pareto_indices)} ({100*len(pareto_indices)/len(X):.1f}%)")
        
        return pareto_indices, pareto_objectives
    
    def rank_by_objective(
        self,
        pareto_indices: np.ndarray,
        pareto_objectives: np.ndarray,
        objective_idx: int,
        ascending: bool = False
    ) -> np.ndarray:
        """
        Rank Pareto-optimal solutions by a specific objective.
        
        Args:
            pareto_indices: Indices of Pareto points
            pareto_objectives: Objective values of Pareto points
            objective_idx: Index of objective to rank by
            ascending: If True, rank from low to high
        
        Returns:
            Ranked indices
        """
        sort_order = np.argsort(pareto_objectives[:, objective_idx])
        if not ascending:
            sort_order = sort_order[::-1]
        
        return pareto_indices[sort_order]



# STRATEGY COMPARISON


class StrategyComparator:
    """
    Compares multiple compound selection strategies.
    
    Example:
        >>> comparator = StrategyComparator(df_test, ['pIC50', 'cost_metric'])
        >>> comparator.add_strategy('Top Potency', lambda df, n: df.nlargest(n, 'pIC50'))
        >>> results = comparator.compare_all(n_select=20)
    """
    
    def __init__(self, df: pd.DataFrame, objective_cols: List[str]):
        """
        Initialize strategy comparator.
        
        Args:
            df: DataFrame with candidate compounds
            objective_cols: Columns representing objectives (e.g., ['pIC50', 'cost_metric'])
        """
        self.df = df.copy()
        self.objective_cols = objective_cols
        self.strategies = {}
    
    def add_strategy(
        self,
        name: str,
        selection_fn: Callable[[pd.DataFrame, int], pd.DataFrame]
    ):
        """
        Register a selection strategy.
        
        Args:
            name: Strategy name
            selection_fn: Function that takes (df, n_select) and returns selected df
        """
        self.strategies[name] = selection_fn
    
    def compare_all(
        self,
        n_select: int,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare all registered strategies.
        
        Args:
            n_select: Number of compounds to select
            metrics: List of metrics to compute (default: all)
        
        Returns:
            DataFrame with comparison results
        """
        metrics = metrics or ['avg_potency', 'avg_cost', 'max_potency', 'trade_off']
        
        results = []
        
        for strategy_name, strategy_fn in self.strategies.items():
            selected_df = strategy_fn(self.df, n_select)
            
            result = {'strategy': strategy_name}
            
            if 'avg_potency' in metrics and 'pIC50' in selected_df.columns:
                result['avg_potency'] = selected_df['pIC50'].mean()
            
            if 'avg_cost' in metrics and 'cost_metric' in selected_df.columns:
                result['avg_cost'] = selected_df['cost_metric'].mean()
            
            if 'max_potency' in metrics and 'pIC50' in selected_df.columns:
                result['max_potency'] = selected_df['pIC50'].max()
            
            if 'trade_off' in metrics:
                if 'pIC50' in selected_df.columns and 'cost_metric' in selected_df.columns:
                    result['trade_off'] = (
                        selected_df['pIC50'].mean() / (selected_df['cost_metric'].mean() + 0.01)
                    )
            
            results.append(result)
        
        return pd.DataFrame(results).sort_values('trade_off', ascending=False)
    
    def visualize_comparison(self, results: pd.DataFrame, output_path: str):
        """
        Create comparison visualizations.
        
        Args:
            results: DataFrame from compare_all()
            output_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Average Potency
        axes[0, 0].barh(results['strategy'], results.get('avg_potency', []), color='steelblue')
        axes[0, 0].set_xlabel('Average pIC50')
        axes[0, 0].set_title('Average Potency by Strategy')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Average Cost
        axes[0, 1].barh(results['strategy'], results.get('avg_cost', []), color='coral')
        axes[0, 1].set_xlabel('Average Cost')
        axes[0, 1].set_title('Average Cost by Strategy')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Plot 3: Max Potency
        axes[1, 0].barh(results['strategy'], results.get('max_potency', []), color='green')
        axes[1, 0].set_xlabel('Maximum pIC50')
        axes[1, 0].set_title('Maximum Potency by Strategy')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Plot 4: Trade-off Score
        axes[1, 1].barh(results['strategy'], results.get('trade_off', []), color='purple')
        axes[1, 1].set_xlabel('Potency/Cost Score')
        axes[1, 1].set_title('Overall Trade-off Score')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {output_path}")



# UTILITY FUNCTIONS


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely convert SMILES to RDKit molecule.
    
    Args:
        smiles: SMILES string
    
    Returns:
        RDKit Mol object or None if invalid
    """
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def calculate_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Calculate comprehensive molecular descriptors.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        Dictionary of descriptors
    """
    if mol is None:
        return {}
    
    try:
        descriptors = {
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'rings': Descriptors.RingCount(mol),
            'tpsa': Descriptors.TPSA(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
        }
        return descriptors
    except:
        return {}


def calculate_synthetic_accessibility(mol: Chem.Mol) -> float:
    """
    Estimate synthetic accessibility score (heuristic).
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        SA score from 1.0 (easy) to 10.0 (hard)
    """
    if mol is None:
        return 10.0
    
    try:
        mw = Descriptors.MolWt(mol)
        rings = Descriptors.RingCount(mol)
        
        try:
            chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        except:
            chiral_centers = 0
        
        try:
            ring_info = mol.GetRingInfo()
            complex_rings = sum(1 for ring in ring_info.AtomRings() if len(ring) > 6)
        except:
            complex_rings = 0
        
        # Heuristic formula
        sa_score = (mw / 500) * 2 + rings * 0.5 + chiral_centers * 1.5 + complex_rings * 2
        return min(10.0, max(1.0, sa_score))
    except:
        return 10.0


def visualize_pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    pareto_indices: np.ndarray,
    output_path: str,
    strategies: Optional[Dict[str, pd.DataFrame]] = None
):
    """
    Visualize Pareto front with optional strategy overlays.
    
    Args:
        df: Full candidate DataFrame
        x_col: Column for x-axis (e.g., 'cost_metric')
        y_col: Column for y-axis (e.g., 'pIC50')
        pareto_indices: Indices of Pareto-optimal points
        output_path: Path to save figure
        strategies: Optional dict of {strategy_name: selected_df}
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all candidates
    ax.scatter(df[x_col], df[y_col],
              alpha=0.3, s=30, c='lightgray', label='All Candidates', zorder=1)
    
    # Plot Pareto-optimal
    pareto_df = df.iloc[pareto_indices]
    ax.scatter(pareto_df[x_col], pareto_df[y_col],
              alpha=0.7, s=100, c='red', marker='*',
              edgecolors='darkred', linewidths=1.5,
              label=f'Pareto-Optimal (n={len(pareto_indices)})', zorder=3)
    
    # Overlay strategies if provided
    if strategies:
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (name, strategy_df) in enumerate(strategies.items()):
            ax.scatter(strategy_df[x_col], strategy_df[y_col],
                      alpha=0.5, s=60, c=colors[i % len(colors)],
                      label=name, zorder=2)
    
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=13)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=13)
    ax.set_title('Multi-Objective Optimization: Pareto Front', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Pareto front plot saved to {output_path}")


def export_compounds_for_ordering(
    df_selected: pd.DataFrame,
    output_path: str,
    include_inchi: bool = True
):
    """
    Export compound list for experimental ordering.
    
    Args:
        df_selected: DataFrame with selected compounds
        output_path: Output CSV path
        include_inchi: Whether to compute InChI identifiers
    """
    export_df = df_selected[[
        'canonical_smiles',
        'pIC50',
        'cost_metric',
        'mol_weight',
        'sa_score'
    ]].copy()
    
    # Add ranking
    export_df.insert(0, 'rank', range(1, len(export_df) + 1))
    
    # Add InChI if requested
    if include_inchi:
        inchis = []
        for smiles in export_df['canonical_smiles']:
            mol = safe_mol_from_smiles(smiles)
            if mol:
                try:
                    inchi = Chem.MolToInchi(mol)
                except:
                    inchi = ''
            else:
                inchi = ''
            inchis.append(inchi)
        export_df.insert(2, 'inchi', inchis)
    
    # Save
    export_df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(export_df)} compounds to {output_path}")



# MODULE METADATA


__version__ = '1.0.0'
__authors__ = ['Amit Chaudhary', 'Ayush Sinha', 'Siddhardh Chochipatla']
__course__ = 'MSML610 - Advanced Machine Learning'
__date__ = 'December 2024'

if __name__ == '__main__':
    print(f"BoTorch Utils Module v{__version__}")
    print(f"Authors: {', '.join(__authors__)}")
    print(f"Course: {__course__}")
    print("\nAvailable classes:")
    print("  - ChemDataProcessor")
    print("  - GPSurrogateModel")
    print("  - MultiObjectiveOptimizer")
    print("  - StrategyComparator")
    print("\nFor usage examples, see BoTorch.API.ipynb")
