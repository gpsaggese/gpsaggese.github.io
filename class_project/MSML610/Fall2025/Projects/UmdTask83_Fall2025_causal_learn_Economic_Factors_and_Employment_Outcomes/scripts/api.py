"""
Main script for running the causal inference pipeline.

This script demonstrates the complete causal inference workflow:
1. Data loading and preprocessing
2. Causal discovery
3. Causal effect estimation
4. Temporal analysis
5. Model comparisons

Usage:
    python3 scripts/api.py
    
Requires:
    - FRED data downloaded (run data/download_data.py first)
    - pip3 install -r requirements.txt
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.utils_data_io import (
    load_economic_data,
    prepare_features_for_causal_discovery,
    get_data_summary
)
from utils.utils_post_processing import (
    discover_causal_structure,
    estimate_causal_effects,
    visualize_causal_graph,
    temporal_effect_estimation,
    plot_temporal_effects
)
from models import RandomForestModel


def main():
    """
    Main function to run the causal inference pipeline.
    """
    print("=" * 70)
    print("Causal Inference Pipeline: Economic Factors and Employment Outcomes")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data from FRED...")
    data_path = os.path.join(project_root, 'data', 'economic_data.csv')
    
    try:
        df = load_economic_data(data_path)
        print(f"  Loaded {len(df)} observations")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    except FileNotFoundError:
        print("  ERROR: Data not found!")
        print("  Run: export FRED_API_KEY=your_key && python3 data/download_data.py")
        return 1
    
    # Define variables for causal analysis
    variables = ['unemployment_rate', 'inflation_rate', 'wage_growth', 
                 'gdp_growth', 'federal_funds_rate']
    
    # Filter to available variables
    available = [v for v in variables if v in df.columns]
    print(f"  Variables: {available}")
    
    # Prepare data for causal discovery
    causal_data = prepare_features_for_causal_discovery(df, available)
    print(f"  Causal data: {causal_data.shape}")
    
    # Step 2: Causal discovery
    print("\n[Step 2] Discovering causal structure (PC algorithm)...")
    causal_graph, edges = discover_causal_structure(
        data=causal_data,
        algorithm='PC',
        alpha=0.05,
        variables=available
    )
    
    print(f"  Discovered {len(edges)} causal relationships:")
    for src, tgt in edges:
        print(f"    {src} -> {tgt}")
    
    # Step 3: Visualize causal graph
    print("\n[Step 3] Visualizing causal DAG...")
    output_dir = os.path.join(project_root, 'outputs', 'causal_graphs')
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_causal_graph(
        graph=causal_graph,
        output_path=os.path.join(output_dir, 'causal_dag.png'),
        title='Causal Structure: Economic Factors'
    )
    print(f"  Saved to: {output_dir}/causal_dag.png")
    
    # Step 4: Causal effect estimation
    print("\n[Step 4] Estimating causal effects...")
    
    effects = {}
    pairs = [
        ('unemployment_rate', 'wage_growth'),
        ('inflation_rate', 'wage_growth'),
        ('federal_funds_rate', 'inflation_rate')
    ]
    
    for treatment, outcome in pairs:
        if treatment in causal_data.columns and outcome in causal_data.columns:
            effect = estimate_causal_effects(
                data=causal_data,
                causal_graph=causal_graph,
                treatment=treatment,
                outcome=outcome,
                method='regression'
            )
            effects[f"{treatment} -> {outcome}"] = effect
            
            sig = "***" if effect['p_value'] < 0.001 else "**" if effect['p_value'] < 0.01 else "*" if effect['p_value'] < 0.05 else ""
            print(f"  {treatment} -> {outcome}: {effect['coefficient']:.4f} {sig}")
    
    # Step 5: Temporal analysis
    print("\n[Step 5] Temporal analysis (rolling windows)...")
    
    if 'unemployment_rate' in df.columns and 'wage_growth' in df.columns:
        temporal = temporal_effect_estimation(
            data=df,
            window_size=36,
            step_size=6,
            treatment='unemployment_rate',
            outcome='wage_growth',
            time_column='date'
        )
        
        print(f"  Analyzed {len(temporal['time'])} time windows")
        print(f"  Effect range: [{temporal['effect'].min():.4f}, {temporal['effect'].max():.4f}]")
        
        output_dir = os.path.join(project_root, 'outputs', 'temporal_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_temporal_effects(
            temporal_results=temporal,
            treatment='unemployment_rate',
            outcome='wage_growth',
            output_path=os.path.join(output_dir, 'temporal_effects.png')
        )
        print(f"  Saved to: {output_dir}/temporal_effects.png")
    
    # Step 6: ML comparison using modular RandomForestModel
    print("\n[Step 6] ML model comparison (Random Forest)...")
    
    features = [c for c in ['unemployment_rate', 'inflation_rate', 'federal_funds_rate'] 
                if c in causal_data.columns]
    target = 'wage_growth'
    
    if target in causal_data.columns and len(features) >= 2:
        X = causal_data[features].dropna()
        y = causal_data.loc[X.index, target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use modular RandomForestModel
        rf_model = RandomForestModel(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train, feature_names=features)
        
        # Evaluate
        metrics = rf_model.evaluate(X_test, y_test)
        print(f"  Random Forest R2: {metrics['r2']:.4f}")
        print(f"  Random Forest RMSE: {metrics['rmse']:.4f}")
        
        # Feature importance using model method
        print("  Feature importance:")
        importance_df = rf_model.get_feature_importance()
        for _, row in importance_df.iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
    
    print("\n" + "=" * 70)
    print("Pipeline complete! Results saved to outputs/")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
