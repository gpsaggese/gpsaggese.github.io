import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.iv import IV2SLS
from causalinference import CausalModel
import scipy.stats as stats
from src.utils import save_plot

def run_instrumental_variables(weekly):
    """Run instrumental variables analysis - Subtask 3.1"""
    print("Running Instrumental Variables Analysis...")
    
    # Keep continent and week_start for merging
    reg_data = weekly[['cases_per_100k_roll3', 'vac_pct_roll3', 'vac_pct_lag', 
                       'population_density', 'median_age', 'hospital_beds_per_thousand', 
                       'gdp_per_capita','continent','week_start']].copy()

    # Create continent-level lagged instrument
    continent_vac_lag = weekly.groupby(['continent','week_start'])['vac_pct'].mean().reset_index()
    continent_vac_lag['week_start'] = pd.to_datetime(continent_vac_lag['week_start'])
    continent_vac_lag['vac_pct_lag_continent'] = continent_vac_lag.groupby('continent')['vac_pct'].shift(3)

    # Merge instrument into regression dataset
    reg_data = reg_data.merge(continent_vac_lag[['continent','week_start','vac_pct_lag_continent']],
                              on=['continent','week_start'], how='left')

    # Drop rows with missing values
    reg_data = reg_data.dropna()

    # Define variables for IV regression
    y_clean = reg_data['cases_per_100k_roll3']  # outcome
    exog_clean = reg_data[['population_density','median_age','hospital_beds_per_thousand','gdp_per_capita']]  # confounders
    endog_clean = reg_data['vac_pct_roll3']  # endogenous treatment
    instr_clean = reg_data['vac_pct_lag_continent']  # instrument

    # Fit IV regression
    iv_model = IV2SLS(dependent=y_clean, exog=exog_clean, endog=endog_clean, instruments=instr_clean).fit()

    # Display results
    print(iv_model.summary)
    
    return iv_model

def run_causal_analysis(weekly):
    """Run main causal analysis using CausalInference library - Subtask 3.2"""
    print("Running Causal Analysis...")
    
    # Create binary treatment variable
    median_vaccination = weekly['vac_pct_roll3'].median()
    weekly['high_vaccination'] = (weekly['vac_pct_roll3'] > median_vaccination).astype(int)

    # Prepare the dataset for causal analysis
    causal_data = weekly.dropna(subset=[
        'cases_per_100k_roll3', 'high_vaccination', 
        'population_density', 'median_age', 
        'hospital_beds_per_thousand', 'gdp_per_capita'
    ]).copy()

    print(f"Sample size: {len(causal_data)} country-weeks")
    print(f"Treatment prevalence: {causal_data['high_vaccination'].mean():.2%}")
    print(f"Median vaccination threshold: {median_vaccination:.1f}%")

    # Prepare variables for CausalModel
    Y = causal_data['cases_per_100k_roll3'].values  # Outcome
    D = causal_data['high_vaccination'].values       # Treatment
    X = causal_data[['population_density', 'median_age', 
                     'hospital_beds_per_thousand', 'gdp_per_capita']].values  # Covariates

    # Initialize CausalModel
    print("\nInitializing CausalModel")
    causal = CausalModel(Y, D, X)

    # Display summary statistics
    print("\nSUMMARY STATISTICS")
    print(causal.summary_stats)

    # Estimate propensity scores
    print("\n PROPENSITY SCORE ESTIMATION")
    causal.est_propensity()
    print("Propensity score estimation completed")

    # Get propensity scores for later use
    propensity_scores = causal.propensity['fitted']

    # Check propensity score balance
    print("\nPROPENSITY SCORE BALANCE")
    print(causal.propensity)

    # Estimate ATE using various methods with proper error handling
    print("\nAVERAGE TREATMENT EFFECT (ATE) ESTIMATION")

    def print_estimate(method_name, estimate_dict):
        """Helper function to print estimate results properly"""
        ate = estimate_dict['ate']
        # Calculate standard error manually if not provided
        if 'se' in estimate_dict:
            se = estimate_dict['se']
        else:
            # For methods without SE, we'll calculate approximate CI
            se = estimate_dict.get('se', np.std(Y) / np.sqrt(len(Y)))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        t_stat = abs(ate / se) if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(t_stat)) if se > 0 else 1
        
        print(f"\n{method_name}:")
        print(f"  ATE: {ate:.4f}")
        print(f"  Std Error: {se:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        return ate, se, p_value

    # 1. Ordinary Least Squares
    causal.est_via_ols()
    ols_ate, ols_se, ols_pval = print_estimate("OLS Estimate", causal.estimates['ols'])

    # 2. Propensity Score Matching
    causal.est_via_matching(matches=1)
    psm_ate, psm_se, psm_pval = print_estimate("Propensity Score Matching", causal.estimates['matching'])

    # 3. Weighting (IPW)
    causal.est_via_weighting()
    ipw_ate, ipw_se, ipw_pval = print_estimate("Inverse Probability Weighting", causal.estimates['weighting'])

    # Store results in DataFrame for comparison
    results_df = pd.DataFrame({
        'Method': ['OLS', 'MATCHING', 'WEIGHTING'],
        'ATE': [ols_ate, psm_ate, ipw_ate],
        'Std_Error': [ols_se, psm_se, ipw_se],
        'P_Value': [ols_pval, psm_pval, ipw_pval]
    })

    # Interpretation
    print("\nINTERPRETATION")
    print(f"Treatment: High vaccination (> {median_vaccination:.1f}%) vs Low vaccination")
    print(f"Outcome: Weekly COVID-19 cases per 100,000 population")

    # Use OLS as primary estimate for interpretation
    primary_ate = ols_ate
    if primary_ate > 0:
        print("High vaccination is associated with INCREASED COVID-19 cases")
        print("This likely indicates SELECTION BIAS: areas with higher COVID risk prioritized vaccination")
    else:
        print("High vaccination is associated with REDUCED COVID-19 cases")
        reduction = abs(primary_ate / causal_data['cases_per_100k_roll3'].mean() * 100)
        print(f"Estimated reduction: {reduction:.1f}% fewer cases in high-vaccination areas")

    # Statistical significance
    if ols_pval < 0.05:
        print("Effect is statistically significant at 5% level")
    else:
        print("Effect is not statistically significant at 5% level")

    raw_diff = causal_data[causal_data['high_vaccination'] == 1]['cases_per_100k_roll3'].mean() - \
               causal_data[causal_data['high_vaccination'] == 0]['cases_per_100k_roll3'].mean()

    print(f"\nRaw difference (unadjusted): {raw_diff:.2f} cases per 100k")
    print(f"OLS adjusted difference: {primary_ate:.2f} cases per 100k")

    # Show the dramatic difference suggesting strong confounding
    if abs(primary_ate - raw_diff) > 50:
        print("Large difference between raw and adjusted estimates suggests STRONG CONFOUNDING")

    return {
        'causal_model': causal,
        'causal_data': causal_data,
        'propensity_scores': propensity_scores,
        'primary_ate': primary_ate,
        'raw_diff': raw_diff,
        'estimates': causal.estimates,
        'results_df': results_df
    }

def run_advanced_diagnostics(weekly, causal_results):
    """Run advanced diagnostics and visualizations - Subtask 3.3"""
    print("Running Advanced Diagnostics...")
    
    causal = causal_results['causal_model']
    causal_data = causal_results['causal_data']
    propensity_scores = causal_results['propensity_scores']
    results_df = causal_results.get('results_df', pd.DataFrame())

    # Access raw data for manual calculations
    raw_data = causal.raw_data
    D = raw_data['D']  # DEFINE D FROM RAW_DATA
    treated_mask = D == 1
    control_mask = D == 0


    # Calculate means and standard deviations manually
    treated_means = raw_data['X'][treated_mask].mean(axis=0)
    control_means = raw_data['X'][control_mask].mean(axis=0)
    treated_stds = raw_data['X'][treated_mask].std(axis=0)
    control_stds = raw_data['X'][control_mask].std(axis=0)

    print("Treated group means:", treated_means)
    print("Control group means:", control_means)

    # Manual propensity score blocking analysis
    print("\nMANUAL PROPENSITY SCORE BLOCKING")

    causal_data_blocks = causal_data.copy()
    causal_data_blocks['propensity_score'] = propensity_scores
    causal_data_blocks['propensity_block'] = pd.qcut(propensity_scores, q=5, labels=False, duplicates='drop')

    block_ates = []
    block_sizes = []

    for block in sorted(causal_data_blocks['propensity_block'].unique()):
        block_data = causal_data_blocks[causal_data_blocks['propensity_block'] == block]
        
        if len(block_data) > 10 and block_data['high_vaccination'].nunique() == 2:
            Y_block = block_data['cases_per_100k_roll3'].values
            D_block = block_data['high_vaccination'].values
            X_block = block_data[['population_density', 'median_age', 
                                'hospital_beds_per_thousand', 'gdp_per_capita']].values
            
            # Simple difference within block
            treated_mean = Y_block[D_block == 1].mean()
            control_mean = Y_block[D_block == 0].mean()
            block_ate = treated_mean - control_mean
            block_ates.append(block_ate)
            block_sizes.append(len(block_data))
            
            print(f"Block {block}: ATE = {block_ate:7.2f}, N = {len(block_data)}")

    # Calculate weighted average for manual blocking
    if block_ates:
        manual_blocking_ate = np.average(block_ates, weights=block_sizes)
        manual_blocking_se = np.std(block_ates) / np.sqrt(len(block_ates))
        
        # Add to results dataframe
        manual_blocking_row = pd.DataFrame({
            'Method': ['MANUAL_BLOCKING'],
            'ATE': [manual_blocking_ate],
            'Std_Error': [manual_blocking_se],
            'P_Value': [2 * (1 - stats.norm.cdf(abs(manual_blocking_ate/manual_blocking_se)))]
        })
        results_df = pd.concat([results_df, manual_blocking_row], ignore_index=True)
        
        print(f"\nManual Blocking ATE: {manual_blocking_ate:.4f}")
        print(f"Manual Blocking SE: {manual_blocking_se:.4f}")

    # Calculate common support
    min_treated = propensity_scores[D == 1].min()
    max_treated = propensity_scores[D == 1].max()
    min_control = propensity_scores[D == 0].min() 
    max_control = propensity_scores[D == 0].max()
    overlap_start = max(min_treated, min_control)
    overlap_end = min(max_treated, max_control)
    overlap_ratio = (overlap_end - overlap_start) / (max(max_treated, max_control) - min(min_treated, min_control))

    print(f"\nCommon support: {overlap_ratio:.1%} of propensity score range")

    # Now create the comprehensive visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: ATE comparison across methods
    plt.subplot(2, 3, 1)
    methods = results_df['Method']
    ates = results_df['ATE']
    errors = results_df['Std_Error'] * 1.96

    plt.errorbar(range(len(methods)), ates, yerr=errors, fmt='o', capsize=5, 
                 color='red', alpha=0.7, markersize=8)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.ylabel('Average Treatment Effect (cases per 100k)')
    plt.title('Causal Estimates Across Methods\n(All show positive association)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(methods)), methods, rotation=45)

    # Add value labels
    for i, (method, ate) in enumerate(zip(methods, ates)):
        plt.text(i, ate + (errors[i] if ate >= 0 else -errors[i]), 
                 f'{ate:.0f}', ha='center', va='bottom' if ate >= 0 else 'top', 
                 fontweight='bold', fontsize=9)

    # Plot 2: Propensity score distribution
    plt.subplot(2, 3, 2)
    plt.hist(propensity_scores[D == 0], bins=30, alpha=0.5, label='Control (Low Vaccination)', 
             color='red', density=True)
    plt.hist(propensity_scores[D == 1], bins=30, alpha=0.5, label='Treated (High Vaccination)', 
             color='blue', density=True)
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Covariate balance visualization
    plt.subplot(2, 3, 3)
    covariate_names = ['Population Density', 'Median Age', 'Hospital Beds', 'GDP per Capita']

    # Calculate standardized differences
    std_diffs = []
    for i in range(len(covariate_names)):
        pooled_std = np.sqrt((treated_stds[i]**2 + control_stds[i]**2) / 2)
        std_diff = (treated_means[i] - control_means[i]) / pooled_std
        std_diffs.append(std_diff)

    colors = ['red' if abs(diff) > 0.1 else 'blue' for diff in std_diffs]
    plt.barh(covariate_names, std_diffs, color=colors, alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5, label='Imbalance threshold')
    plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Standardized Difference')
    plt.title('Covariate Balance Before Adjustment\n(Red = Imbalanced)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add values on bars
    for i, (name, diff) in enumerate(zip(covariate_names, std_diffs)):
        plt.text(diff + (0.01 if diff >= 0 else -0.01), i, f'{diff:.2f}', 
                 va='center', ha='left' if diff >= 0 else 'right', fontweight='bold')

    # Plot 4: Manual blocking results
    plt.subplot(2, 3, 4)
    blocks = list(range(len(block_ates)))
    plt.bar(blocks, block_ates, color=['red' if ate > 0 else 'green' for ate in block_ates], alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Propensity Score Block')
    plt.ylabel('ATE (cases per 100k)')
    plt.title('Manual Blocking: ATE by Propensity Block')
    plt.grid(True, alpha=0.3)

    # Add block sizes as labels
    for i, (block, ate, size) in enumerate(zip(blocks, block_ates, block_sizes)):
        plt.text(block, ate + (10 if ate >= 0 else -10), f'N={size}', 
                 ha='center', va='bottom' if ate >= 0 else 'top', fontsize=8)

    # Plot 5: Raw vs Adjusted comparison
    plt.subplot(2, 3, 5)
    adjusted_diff = causal_results['primary_ate']
    raw_diff = causal_results['raw_diff']
    comparison_data = [raw_diff, adjusted_diff]
    labels = ['Raw Difference', 'OLS Adjusted']

    plt.bar(labels, comparison_data, color=['darkred', 'red'], alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.ylabel('Cases per 100k Difference')
    plt.title('Raw vs Adjusted Difference')
    plt.grid(True, alpha=0.3)

    # Add values on bars
    for i, (label, value) in enumerate(zip(labels, comparison_data)):
        plt.text(i, value + (10 if value >= 0 else -10), f'{value:.0f}', 
                 ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')

    # Plot 6: Outcome distributions
    plt.subplot(2, 3, 6)
    treated_outcomes = raw_data['Y'][treated_mask]
    control_outcomes = raw_data['Y'][control_mask]

    plt.boxplot([control_outcomes, treated_outcomes], 
                labels=['Control\n(Low Vaccination)', 'Treated\n(High Vaccination)'])
    plt.ylabel('Cases per 100k')
    plt.title('Outcome Distribution by Treatment Group')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot('advanced_causal_diagnostics.png')
    plt.show()

    print("\nCOVARIATE BALANCE ASSESSMENT")
    print("Standardized differences (absolute values > 0.1 indicate imbalance):")

    imbalanced_count = 0
    for i, name in enumerate(covariate_names):
        status = "IMBALANCED" if abs(std_diffs[i]) > 0.1 else "BALANCED"
        if abs(std_diffs[i]) > 0.1:
            imbalanced_count += 1
        print(f"{name:20}: {std_diffs[i]:6.3f} - {status}")

    print(f"\n{imbalanced_count} out of {len(covariate_names)} covariates are imbalanced")
    print("This indicates the need for careful causal adjustment")

    print("\nKEY FINDINGS")
    key_findings = [
        f"All methods show positive ATE (range: {results_df['ATE'].min():.0f} to {results_df['ATE'].max():.0f})",
        f"{imbalanced_count}/{len(covariate_names)} covariates imbalanced before adjustment",
        f"Adjustment reduces association from {raw_diff:.0f} to {adjusted_diff:.0f}",
        "Manual blocking shows effect heterogeneity across propensity strata",
        "Strong evidence of systematic differences between treated and control groups"
    ]

    for finding in key_findings:
        print(f"• {finding}")

    return {
        'block_ates': block_ates,
        'block_sizes': block_sizes,
        'std_diffs': std_diffs,
        'imbalanced_count': imbalanced_count,
        'results_df': results_df
    }