**Description**

JAX is a high-performance machine learning library designed for numerical computing, enabling automatic differentiation and GPU/TPU acceleration. It provides a flexible and composable approach to building machine learning models, allowing for rapid experimentation and optimization.

Technologies Used
JAX

- Offers automatic differentiation for functions, enabling gradient-based optimization.
- Supports Just-In-Time (JIT) compilation for accelerated performance on CPUs, GPUs, and TPUs.
- Facilitates vectorization and parallelization of operations, making it suitable for large-scale computations.

---

### Project 1: Predicting House Prices (Difficulty: 1 - Easy)

**Project Objective**  
The goal of this project is to develop a regression model that predicts house prices based on various features such as size, location, and number of rooms. Students will optimize the model to minimize prediction error.

**Dataset Suggestions**  
- **Dataset**: California Housing Prices Dataset  
- **Source**: Available on Kaggle [California Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

**Tasks**  
- Data Preprocessing:  
  Clean the dataset by handling missing values and normalizing features.

- Feature Engineering:  
  Create additional features like price per square foot or age of the house.

- Model Development:  
  Use JAX to implement a linear regression model and optimize parameters using gradient descent.

- Model Evaluation:  
  Evaluate model performance using metrics such as Mean Absolute Error (MAE) and R-squared.

- Visualization:  
  Visualize the relationship between predicted and actual house prices using Matplotlib.

---

### Project 2: Image Classification with Convolutional Neural Networks (Difficulty: 2 - Medium)

**Project Objective**  
Students will build a convolutional neural network (CNN) to classify images from a dataset of handwritten digits (MNIST). The objective is to achieve high accuracy in recognizing digits while optimizing the model architecture.

**Dataset Suggestions**  
- **Dataset**: MNIST Handwritten Digits  
- **Source**: Available on Kaggle [MNIST Handwritten Digits](https://www.kaggle.com/c/digit-recognizer/data)

**Tasks**  
- Data Loading:  
  Load and preprocess the MNIST dataset, including normalization and reshaping of image data.

- Model Architecture:  
  Design a CNN architecture using JAX, including convolutional layers, activation functions, and pooling layers.

- Training the Model:  
  Train the CNN using stochastic gradient descent and JAX's automatic differentiation for backpropagation.

- Model Evaluation:  
  Assess model performance using accuracy metrics and confusion matrix.

- Hyperparameter Tuning:  
  Experiment with different hyperparameters (learning rate, batch size) to optimize model performance.

---

### Project 3: Time Series Forecasting with LSTM (Difficulty: 3 - Hard)

**Project Objective**  
This project focuses on building a Long Short-Term Memory (LSTM) model to forecast future stock prices based on historical data. The objective is to capture temporal dependencies and improve prediction accuracy.

**Dataset Suggestions**  
- **Dataset**: Historical Stock Prices  
- **Source**: Yahoo Finance API (free tier) for stock data (e.g., Apple Inc. - AAPL)

**Tasks**  
- Data Collection:  
  Use the Yahoo Finance API to fetch historical stock price data for a specified period.

- Data Preprocessing:  
  Clean the data by handling missing values and transforming the time series into a supervised learning format.

- LSTM Model Development:  
  Implement an LSTM model using JAX to capture time dependencies in the stock prices.

- Model Training:  
  Train the LSTM model with appropriate loss functions and optimizers, utilizing JAX's JIT compilation for performance.

- Model Evaluation:  
  Evaluate the model using metrics such as Mean Squared Error (MSE) and visualize the predicted vs. actual stock prices.

**Bonus Ideas (Optional)**  
- Experiment with different LSTM architectures (e.g., stacked LSTMs).
- Integrate additional features like trading volume or technical indicators for improved forecasting.
- Compare LSTM performance with traditional time series models (e.g., ARIMA).

