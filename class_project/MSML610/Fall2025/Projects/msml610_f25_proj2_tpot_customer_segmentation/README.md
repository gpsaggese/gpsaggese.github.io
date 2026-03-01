# MSML610 Project 2 – TPOT-based Customer Segmentation

## Objective
Use the UCI Online Retail dataset to build an end-to-end machine learning pipeline that:
- Cleans transactional data
- Aggregates customers into behavioral features
- Uses TPOT (AutoML) to learn an optimal preprocessing pipeline
- Applies clustering to derive interpretable customer segments

## Dataset
Online Retail Data Set (UCI Machine Learning Repository):

Users must download `Online Retail.xlsx` separately and place it in the `data/` folder.


## Data Cleaning
The cleaning pipeline:

- Converts dates to proper datetime format  
- Removes missing `CustomerID` values  
- Removes cancelled orders (Invoice numbers starting with “C”)  
- Filters out negative or zero quantities/prices  
- Adds `LineTotal = Quantity × UnitPrice`  
- Removes extreme outliers using 1st–99th percentile trimming  

All logic is implemented in:  
`src/features.py`

---

## Feature Engineering
Customer-level features are aggregated from transaction data, including:

- Number of invoices  
- Total items purchased  
- Unique products purchased  
- Total spend  
- Average unit price  
- Average basket size  
- First & last purchase timestamps  
- **Recency (days since last purchase)**  
- **Tenure (time between first and last purchase)**  
- **Mean inter-purchase interval**

These features form the input matrix for clustering.

---

## TPOT-Based Preprocessing (AutoML)
TPOT is used to automatically learn:

- Scaling  
- Normalization  
- Polynomial transforms  
- Optimal preprocessing steps  

The final preprocessing pipeline is extracted (estimator removed) and stored at: outputs/tpot_preprocess_transformer.joblib

## Clustering
This project evaluates multiple clustering configurations:

- **K-Means** (k = 2 to 10)
- **Gaussian Mixture Models** (GMM)

Each model is scored using:

- Silhouette Score  
- Calinski-Harabasz Index  
- Davies-Bouldin Index  

The best model is selected automatically.

Results are saved to: outputs/clustering_search.csv

## Visualization
A 2-D PCA projection is generated to visualize cluster separations: outputs/clusters_pca.png

Each point represents a customer; colors represent segment assignments.



## Segment Output
Final customer segments are written to:outputs/customer_segments.csv

The script also assigns human-interpretable segment labels such as:

- High-Value Customers  
- Mid-Tier Customers  
- Low-Value / At-Risk  

A text summary is generated in:outputs/REPORT.txt


## How to Run

```bash
pip install -r requirements.txt
python -m src.main


