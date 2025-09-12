**Description**

AutoGluon is an open-source AutoML framework designed to simplify the process of building machine learning models. It automates tasks such as data preprocessing, model selection, hyperparameter tuning, and ensembling, making it accessible for both beginners and experienced practitioners. With its ability to handle various data types and tasks, AutoGluon allows users to quickly prototype and deploy robust models with minimal effort.

Technologies Used
AutoGluon

- Provides automated machine learning capabilities for structured data, text, and image data.
- Supports a wide range of ML tasks including classification, regression, and object detection.
- Offers easy-to-use APIs for model training, evaluation, and prediction.

---

### Project 1: Predicting House Prices
**Difficulty**: 1 (Easy)

**Project Objective**: The goal is to predict house prices based on various features such as location, size, number of rooms, and amenities. Students will optimize the model to achieve the lowest possible mean absolute error (MAE).

**Dataset Suggestions**: Use the "Ames Housing Dataset" available on Kaggle. It contains detailed information about houses sold in Ames, Iowa.

**Tasks**:
- **Data Loading**: Import the Ames Housing Dataset into a Pandas DataFrame.
- **Data Preprocessing**: Handle missing values and categorical variables using AutoGluon’s built-in methods.
- **Model Training**: Use AutoGluon to train multiple regression models on the dataset.
- **Model Evaluation**: Assess model performance using MAE and visualize results with Matplotlib.
- **Feature Importance**: Analyze which features were most influential in predicting house prices.

---

### Project 2: Customer Churn Prediction
**Difficulty**: 2 (Medium)

**Project Objective**: The aim is to predict whether customers will churn based on their usage patterns and demographics. The project will focus on optimizing the model for accuracy and recall to minimize false negatives.

**Dataset Suggestions**: Use the "Telco Customer Churn" dataset available on Kaggle, which contains customer information and their subscription status.

**Tasks**:
- **Data Loading**: Import the Telco Customer Churn dataset into a Pandas DataFrame.
- **Data Preprocessing**: Clean the data by handling missing values, encoding categorical features, and normalizing numerical features.
- **Model Training**: Utilize AutoGluon to automatically train and tune various classification models.
- **Performance Metrics**: Evaluate model performance using accuracy, recall, and ROC-AUC scores.
- **Model Interpretation**: Use SHAP or LIME to interpret model predictions and identify key features influencing churn.

---

### Project 3: Image Classification of Handwritten Digits
**Difficulty**: 3 (Hard)

**Project Objective**: The goal is to classify images of handwritten digits (0-9) using a dataset of images. Students will optimize the model for accuracy and explore the impact of different architectures.

**Dataset Suggestions**: Use the "MNIST Handwritten Digits" dataset available on Kaggle, which contains 70,000 images of handwritten digits.

**Tasks**:
- **Data Loading**: Import the MNIST dataset using AutoGluon’s image data handling capabilities.
- **Data Augmentation**: Implement data augmentation techniques to enhance the training dataset and improve model generalization.
- **Model Training**: Utilize AutoGluon to train convolutional neural networks (CNNs) and compare their performance against traditional models.
- **Hyperparameter Tuning**: Experiment with hyperparameter tuning to optimize the CNN architectures for better accuracy.
- **Evaluation and Visualization**: Evaluate the model using confusion matrices and visualize misclassified images to understand model limitations.

**Bonus Ideas (Optional)**: 
- For Project 1, attempt to implement feature engineering techniques to improve model performance.
- For Project 2, explore ensemble methods to combine predictions from multiple models for better accuracy.
- For Project 3, investigate transfer learning by using pre-trained models on similar tasks to enhance classification performance.

