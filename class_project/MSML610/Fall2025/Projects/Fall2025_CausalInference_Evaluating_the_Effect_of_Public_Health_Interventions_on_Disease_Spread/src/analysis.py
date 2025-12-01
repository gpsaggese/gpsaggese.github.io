import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import save_plot

def exploratory_analysis(weekly):
    """Perform exploratory data analysis with visualizations"""
    
    # 1. Correlation matrix
    key_vars = ['vac_pct','vac_pct_lag','cases_per_100k','cases_per_100k_lag',
                'cases_per_100k_roll3','deaths_per_100k','deaths_per_100k_roll3']
    available_vars = [var for var in key_vars if var in weekly.columns]
    
    plt.figure(figsize=(10, 8))
    corr = weekly[available_vars].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Key Variables')
    save_plot('correlation_matrix.png')
    plt.show()
    
    # 2. Scatter plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(weekly['vac_pct_lag'], weekly['cases_per_100k_roll3'], alpha=0.6, s=10)
    plt.xlabel('Vaccination % (lag 3 weeks)')
    plt.ylabel('Cases per 100k (3-week rolling)')
    plt.title('Lagged Vaccination vs Cases')
    
    plt.subplot(1, 2, 2)
    plt.scatter(weekly['vac_pct_lag'], weekly['deaths_per_100k_roll3'], alpha=0.6, s=10)
    plt.xlabel('Vaccination % (lag 3 weeks)')
    plt.ylabel('Deaths per 100k (3-week rolling)')
    plt.title('Lagged Vaccination vs Deaths')
    
    plt.tight_layout()
    save_plot('scatter_plots.png')
    plt.show()
    
    print("Exploratory analysis completed")

def plot_country_trajectories(weekly, countries, variable='cases_per_100k'):
    """Plot variable trajectories for selected countries"""
    plt.figure(figsize=(12, 6))
    
    for country in countries:
        country_data = weekly[weekly['country_code'] == country]
        if len(country_data) > 0:
            plt.plot(country_data['week_start'], country_data[variable], 
                    label=country, linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel(variable.replace('_', ' ').title())
    plt.title(f'{variable.replace("_", " ").title()} Over Time by Country')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(f'country_trajectories_{variable}.png')
    plt.show()

def plot_country_trajectories_combined(weekly, countries):
    """Plot combined vaccination and COVID trends for selected countries - Subtask 2.4"""
    for country in countries:
        tmp = weekly[weekly['country_code'] == country].sort_values('week_start')
        plt.figure(figsize=(10,4))
        plt.plot(tmp['week_start'], tmp['vac_pct'], label='Vaccination %', marker='o')
        plt.plot(tmp['week_start'], tmp['cases_per_100k'], label='Cases per 100k', marker='x')
        plt.plot(tmp['week_start'], tmp['deaths_per_100k'], label='Deaths per 100k', marker='^')
        plt.title(f'Weekly Vaccination & COVID Trends: {country}')
        plt.xlabel('Week Start')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

def plot_continent_trends(weekly):
    """Plot vaccination trends by continent - Subtask 2.5"""
    continent_vac = weekly.groupby(['continent','week_start'])['vac_pct'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=continent_vac, x='week_start', y='vac_pct', hue='continent', marker='o', linewidth=2)
    plt.title('Vaccination % Over Time by Continent')
    plt.xlabel('Week Start')
    plt.ylabel('Average Vaccination %')
    plt.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot('vaccination_by_continent.png')
    plt.show()
    
    print("Continental vaccination trends plotted")

def plot_top_countries(weekly):
    """Plot top 5 countries by vaccination % - Subtask 2.5"""
    # Top 5 countries by maximum vaccination %
    top_countries = weekly.groupby('country_code')['vac_pct'].max().sort_values(ascending=False).head(5).index
    top_vac = weekly[weekly['country_code'].isin(top_countries)]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=top_vac, x='week_start', y='vac_pct', hue='country_code', marker='o', linewidth=2)
    plt.title('Top 5 Countries by Maximum Vaccination %')
    plt.xlabel('Week Start')
    plt.ylabel('Vaccination %')
    plt.legend(title='Country Code', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot('top_countries_vaccination.png')
    plt.show()
    
    print("Top countries vaccination trends plotted")

def confounder_exploration(weekly):
    """Comprehensive confounder exploration - Subtask 2.6"""
    confounders = ['population_density', 'median_age', 'hospital_beds_per_thousand', 'gdp_per_capita']
    outcome_vars = ['vac_pct', 'cases_per_100k_roll3', 'deaths_per_100k_roll3']
    
    print("Creating confounder exploration plots...")
    
    # Scatterplots of confounders vs vaccination and outcomes
    for conf in confounders:
        for var in outcome_vars:
            if conf in weekly.columns and var in weekly.columns:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=weekly, x=conf, y=var, alpha=0.6, s=30)
                plt.title(f'{var.replace("_", " ").title()} vs {conf.replace("_", " ").title()}')
                plt.xlabel(conf.replace('_',' ').title())
                plt.ylabel(var.replace('_',' ').title())
                
                # Add correlation coefficient
                correlation = weekly[conf].corr(weekly[var])
                plt.annotate(f'Correlation: {correlation:.3f}', 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                save_plot(f'{var}_vs_{conf}.png')
                plt.show()
    
    # Comprehensive correlation matrix
    corr_vars = outcome_vars + confounders
    available_corr_vars = [var for var in corr_vars if var in weekly.columns]
    
    if available_corr_vars:
        corr_matrix = weekly[available_corr_vars].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                   square=True, cbar_kws={"shrink": .8})
        plt.title('Comprehensive Correlation Matrix: Confounders, Vaccination & Outcomes')
        plt.tight_layout()
        save_plot('comprehensive_correlation_matrix.png')
        plt.show()
        
        print("Comprehensive correlation matrix created")
    
    # Distribution of confounders
    plt.figure(figsize=(15, 10))
    for i, conf in enumerate(confounders, 1):
        if conf in weekly.columns:
            plt.subplot(2, 2, i)
            weekly[conf].hist(bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {conf.replace("_", " ").title()}')
            plt.xlabel(conf.replace('_', ' ').title())
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    save_plot('confounder_distributions.png')
    plt.show()
    
    print("Confounder exploration completed")

def summary_statistics(weekly):
    """Generate summary statistics for key variables"""
    key_vars = ['vac_pct', 'cases_per_100k', 'deaths_per_100k']
    available_vars = [var for var in key_vars if var in weekly.columns]
    return weekly[available_vars].describe()