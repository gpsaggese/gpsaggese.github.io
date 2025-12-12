# Optuna Customer Segmentation 

## Business Problem
Segment 4,338 customers from an online retailer using RFM analysis and Optuna-optimized clustering.

## Solution Overview

### Step 1: Data Preparation
Load 392,692 transactions from UCI Online Retail dataset

### Step 2: Feature Engineering (RFM)
- Recency (R): Days since last purchase
- Frequency (F): Number of transactions  
- Monetary (M): Total spending

### Step 3: Preprocessing
Scale features using StandardScaler

### Step 4: Hyperparameter Optimization
Optuna tests 100 different clustering configurations

### Step 5: Clustering Results
5 optimal customer clusters identified

## Expected Results

- Silhouette Score: 0.49 (Good)
- Davies-Bouldin Index: 1.23 (Acceptable)

## Business Insights
- Total Revenue: $9.7M
- VIP Cluster: 35% of revenue
- Potential uplift: 41%
