**Description**

Haiku is a powerful tool designed for creating and managing machine learning workflows with a focus on simplicity and collaboration. It allows users to build, train, and deploy models efficiently while providing a clear structure for versioning and experimentation. Key features include:

- **Workflow Management**: Streamlines the process of creating and managing ML workflows.
- **Version Control**: Keeps track of model versions and experiment results.
- **Collaboration**: Facilitates teamwork through shared projects and insights.
- **Integration**: Easily connects with popular data sources and libraries.

---

### Project 1: Predicting House Prices (Difficulty: 1)

**Project Objective**  
The goal is to predict house prices based on various features such as location, size, and amenities. Students will optimize their models to achieve the lowest possible Mean Absolute Error (MAE).

**Dataset Suggestions**  
- **Dataset**: "House Prices - Advanced Regression Techniques" (Kaggle)  
- **Link**: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Tasks**  
- **Data Ingestion**: Load the dataset into a Pandas DataFrame using Haiku's data management features.  
- **Data Preprocessing**: Handle missing values and categorical variables using appropriate techniques.  
- **Feature Engineering**: Create new features that may help improve the model's performance.  
- **Model Selection**: Train multiple regression models (e.g., Linear Regression, Random Forest) and evaluate their performance.  
- **Model Evaluation**: Use metrics like MAE to assess model accuracy and select the best-performing model.  
- **Deployment**: Utilize Haiku to deploy the final model for predictions.

### Bonus Ideas (Optional)  
- Compare model performance using different feature sets.  
- Implement cross-validation to ensure robust model evaluation.  

---

### Project 2: Customer Segmentation Using Clustering (Difficulty: 2)

**Project Objective**  
The goal is to segment customers based on purchasing behavior to identify distinct groups. Students will optimize clustering algorithms to maximize intra-cluster similarity and minimize inter-cluster similarity.

**Dataset Suggestions**  
- **Dataset**: "Online Retail Data Set" (UCI Machine Learning Repository)  
- **Link**: [UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

**Tasks**  
- **Data Ingestion**: Load the dataset and clean it to remove any irrelevant or missing entries.  
- **Exploratory Data Analysis**: Perform EDA to understand customer purchasing patterns and visualize the data.  
- **Feature Selection**: Identify relevant features (e.g., total purchase amount, frequency of purchases) for clustering.  
- **Clustering**: Apply clustering algorithms (e.g., K-Means, DBSCAN) to segment customers and evaluate clusters using silhouette scores.  
- **Visualization**: Visualize the clusters using techniques like PCA or t-SNE to understand customer segments.  
- **Insights Generation**: Generate actionable insights based on the identified customer segments.

### Bonus Ideas (Optional)  
- Experiment with different clustering algorithms to compare results.  
- Analyze customer segments over time to understand trends.  

---

### Project 3: Sentiment Analysis on Product Reviews (Difficulty: 3)

**Project Objective**  
The goal is to perform sentiment analysis on product reviews to classify them as positive, negative, or neutral. Students will optimize their models for accuracy and F1-score.

**Dataset Suggestions**  
- **Dataset**: "Amazon Product Reviews" (Kaggle)  
- **Link**: [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

**Tasks**  
- **Data Ingestion**: Load and preprocess the review dataset, focusing on text data cleaning and tokenization.  
- **Text Vectorization**: Convert text data into numerical format using techniques like TF-IDF or Word Embeddings.  
- **Model Training**: Train various classification models (e.g., Logistic Regression, LSTM) and fine-tune hyperparameters for better performance.  
- **Model Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, and F1-score.  
- **Error Analysis**: Perform error analysis to identify misclassifications and improve the model.  
- **Deployment**: Deploy the trained sentiment analysis model using Haiku for real-time predictions on new reviews.

### Bonus Ideas (Optional)  
- Implement a dashboard to visualize sentiment trends over time.  
- Compare the performance of traditional ML models with deep learning approaches.

