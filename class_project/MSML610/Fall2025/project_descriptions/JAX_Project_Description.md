**Description**

JAX is a high-performance machine learning library designed for numerical computing, offering automatic differentiation, GPU/TPU acceleration, and a NumPy-like API. It enables flexible and composable model development while supporting functional programming paradigms. With features like JIT compilation and vectorization, JAX is particularly well-suited for building scalable and efficient machine learning models.

**Technologies Used**  
JAX  
- Provides automatic differentiation for gradient-based optimization.  
- Supports Just-In-Time (JIT) compilation for accelerated performance on CPUs, GPUs, and TPUs.  
- Facilitates vectorization and parallelization for large-scale computations.  
- Uses a NumPy-like API, making it intuitive for scientific computing.  

---

### Project 1: Predicting Bike Sharing Demand (Difficulty: 1 - Easy)  

**Project Objective**  
Build a regression model to predict daily bike rental counts based on weather and seasonal factors. Use JAX for model development and gradient-based optimization.  

**Dataset Suggestions**  
- **Dataset**: [Bike Sharing Demand Dataset](https://www.kaggle.com/datasets/c/bike-sharing-demand)  
- **Domain**: Transportation, tabular time-related regression.  

**Tasks**  
- **Data Preprocessing**: Handle missing values, extract datetime features (day, season, weather).  
- **Feature Engineering**: Create features like holiday/weekend flags or temperature bins.  
- **Model Development**: Implement linear regression and polynomial regression in JAX.  
- **Model Evaluation**: Use RMSE and R² to evaluate predictions.  
- **Visualization**: Plot predicted vs. actual bike rental demand.  

---

### Project 2: Wildlife Image Classification (Difficulty: 2 - Medium)  

**Project Objective**  
Develop a CNN to classify animal species from camera-trap images. Use JAX for model development and optimize hyperparameters for improved accuracy.  

**Dataset Suggestions**  
- **Dataset**: [Animals-10 Image Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) (10 categories of animals).  
- **Domain**: Image classification.  

**Tasks**  
- **Data Loading**: Load and preprocess images (resize, normalize).  
- **CNN Architecture**: Implement convolutional and pooling layers in JAX.  
- **Training the Model**: Use SGD/Adam optimizers with JAX’s autodiff for backpropagation.  
- **Hyperparameter Tuning**: Experiment with kernel sizes, dropout, and learning rates.  
- **Model Evaluation**: Use accuracy, precision, recall, and confusion matrices.  
- **Visualization**: Display examples of correct vs. misclassified images.  

---

### Project 3: Retail Sales Forecasting with LSTMs (Difficulty: 3 - Hard)  

**Project Objective**  
Build an LSTM-based model in JAX to forecast retail sales at the store and product level. Capture multiple seasonalities (weekly, yearly) and event effects (holidays, promotions) to improve forecasting accuracy.  

**Dataset Suggestions**  
- **Dataset**: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)  
- **Domain**: Retail sales forecasting.  

**Tasks**  
- **Data Preprocessing**: Clean and aggregate sales data, create time windows, and encode holiday/promotional events.  
- **Model Development**: Implement LSTM and GRU architectures in JAX for sequence modeling.  
- **Training**: Train models using JIT-compiled optimization in JAX for performance gains.  
- **Evaluation**: Compare results using MAE, RMSE, and MAPE across stores and product categories.  
- **Visualization**: Plot actual vs. predicted sales, highlighting holiday and promotion effects.  
- **Advanced**: Extend to multivariate forecasting by including external regressors like oil prices or inflation.  

---
