**Description**

In this project, students will utilize auto-sklearn, an automated machine learning toolkit for Python, to streamline the process of model selection and hyperparameter optimization. Auto-sklearn automatically searches for the best machine learning algorithms and their configurations, enabling students to focus on data preparation and analysis rather than manual tuning. The tool is designed for supervised classification and regression tasks, making it versatile for various datasets.

Technologies Used
auto-sklearn

- Automates the process of model selection and hyperparameter tuning.
- Supports a wide range of machine learning algorithms from scikit-learn.
- Provides ensemble methods to improve predictive performance.

### Project 1: Predicting Housing Prices (Difficulty: 1)

**Project Objective**  
Develop a predictive model to estimate housing prices based on various features such as location, size, and number of bedrooms. The goal is to optimize the accuracy of the price predictions.

**Dataset Suggestions**  
- Use the "Ames Housing Dataset" available on Kaggle: [Ames Housing Dataset](https://www.kaggle.com/datasets/prestonvong/AmesHousing)  

**Tasks**  
- Data Preprocessing:
  - Clean the dataset by handling missing values and encoding categorical variables.
  
- Feature Selection:
  - Identify relevant features that significantly impact housing prices using correlation analysis.
  
- Model Training with auto-sklearn:
  - Set up auto-sklearn to automatically select the best model and hyperparameters for regression.
  
- Model Evaluation:
  - Evaluate model performance using metrics such as RMSE (Root Mean Squared Error) and R² score.
  
- Visualization:
  - Visualize the predicted vs. actual prices using scatter plots.

### Project 2: Customer Segmentation (Difficulty: 2)

**Project Objective**  
Segment customers based on purchasing behavior to identify distinct groups for targeted marketing strategies. The objective is to optimize clustering results to enhance marketing effectiveness.

**Dataset Suggestions**  
- Use the "Online Retail Dataset" available on UCI Machine Learning Repository: [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)  

**Tasks**  
- Data Cleaning:
  - Preprocess the dataset by removing duplicates and handling missing values.
  
- Feature Engineering:
  - Create features such as total purchase amount and frequency of purchases for each customer.
  
- Clustering with auto-sklearn:
  - Implement auto-sklearn to identify the best clustering algorithms and hyperparameters for customer segmentation.
  
- Model Evaluation:
  - Evaluate the clustering results using silhouette score and Davies-Bouldin index.
  
- Visualization:
  - Visualize clusters using PCA or t-SNE to reduce dimensionality.

### Project 3: Predicting Customer Churn (Difficulty: 3)

**Project Objective**  
Build a predictive model to identify customers likely to churn based on their interaction and usage data. The goal is to optimize the model's ability to predict churn effectively.

**Dataset Suggestions**  
- Use the "Telco Customer Churn" dataset available on Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

**Tasks**  
- Data Preprocessing:
  - Clean the dataset by addressing missing values and encoding categorical features.
  
- Feature Engineering:
  - Create new features that capture customer engagement metrics, such as tenure and service usage.
  
- Classification with auto-sklearn:
  - Utilize auto-sklearn to automatically select the best classification model and hyperparameters for churn prediction.
  
- Model Evaluation:
  - Assess model performance using metrics like accuracy, precision, recall, and F1 score.
  
- Visualization:
  - Visualize feature importance and confusion matrix to analyze model performance.

**Bonus Ideas (Optional)**  
- Implement cross-validation techniques to improve model robustness.
- Compare results with traditional manual tuning approaches to highlight the efficiency of auto-sklearn.
- Explore ensemble methods to further enhance predictive performance.

