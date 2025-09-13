**Description**

In this project, students will utilize TensorFlow Probability, a library for probabilistic reasoning and statistical analysis in TensorFlow, to create models that incorporate uncertainty in predictions. This tool enables users to build complex probabilistic models, perform variational inference, and sample from distributions, making it ideal for projects that require understanding and modeling uncertainty in data.

---

### Project 1: Predicting House Prices with Uncertainty (Difficulty: 1)

**Project Objective**: Develop a model to predict house prices while quantifying the uncertainty associated with those predictions.

**Dataset Suggestions**: 
- Use the "Ames Housing Dataset" available on Kaggle ([Ames Housing Dataset](https://www.kaggle.com/datasets/prestonvong/Ames-Housing-Data)).

**Tasks**:
- **Data Preprocessing**: Clean and preprocess the dataset, handling missing values and categorical variables.
- **Model Development**: Build a probabilistic regression model using TensorFlow Probability to predict house prices.
- **Uncertainty Quantification**: Implement methods to quantify uncertainty in predictions (e.g., credible intervals).
- **Model Evaluation**: Assess the model's performance using metrics like MAE and visualize prediction intervals.
- **Visualization**: Create visualizations to show predicted prices alongside uncertainty bands.

---

### Project 2: Time Series Forecasting with Probabilistic Models (Difficulty: 2)

**Project Objective**: Create a model to forecast future values in a time series dataset while accounting for the inherent uncertainty in predictions.

**Dataset Suggestions**: 
- Use the "Air Quality" dataset available on Kaggle ([Air Quality Dataset](https://www.kaggle.com/datasets/uciml/air-quality-uci)).

**Tasks**:
- **Data Preparation**: Clean the dataset, focusing on the time series aspect and handling missing values.
- **Exploratory Data Analysis**: Analyze seasonality and trends in the time series data.
- **Probabilistic Modeling**: Build a Gaussian Process model using TensorFlow Probability to forecast future air quality levels.
- **Uncertainty Estimation**: Incorporate uncertainty estimates into the forecasts and visualize confidence intervals.
- **Performance Evaluation**: Evaluate the model's predictions against actual values using RMSE and visualize the forecast against observed data.

---

### Project 3: Bayesian Neural Networks for Image Classification (Difficulty: 3)

**Project Objective**: Implement a Bayesian Neural Network (BNN) to classify images while providing uncertainty estimates for each prediction.

**Dataset Suggestions**: 
- Use the "CIFAR-10" dataset available through TensorFlow Datasets ([CIFAR-10 Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/cifar10)).

**Tasks**:
- **Data Loading**: Load the CIFAR-10 dataset and preprocess the images for training.
- **Model Architecture**: Design a Bayesian Neural Network using TensorFlow Probability to classify the images.
- **Training with Uncertainty**: Train the BNN, employing variational inference to estimate posterior distributions over the weights.
- **Prediction and Uncertainty**: Make predictions on test images and quantify uncertainty using predictive distributions.
- **Evaluation and Visualization**: Evaluate classification accuracy and visualize uncertainty in the predictions using confidence maps.

**Bonus Ideas (Optional)**:
- For Project 1, compare the probabilistic model with a standard regression model to highlight differences in uncertainty quantification.
- For Project 2, explore different time series models (e.g., ARIMA vs. Gaussian Processes) and analyze their performance.
- For Project 3, experiment with different architectures (e.g., convolutional layers) and assess how they impact uncertainty estimates in predictions.

