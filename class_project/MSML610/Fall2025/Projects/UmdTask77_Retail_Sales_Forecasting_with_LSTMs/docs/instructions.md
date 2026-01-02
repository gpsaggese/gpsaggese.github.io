**Description**

JAX is a high-performance machine learning library designed for numerical computing, offering automatic differentiation, GPU/TPU acceleration, and a NumPy-like API. It enables flexible and composable model development while supporting functional programming paradigms. With features like JIT compilation and vectorization, JAX is particularly well-suited for building scalable and efficient machine learning models.

**Technologies Used**  
JAX  
- Provides automatic differentiation for gradient-based optimization.  
- Supports Just-In-Time (JIT) compilation for accelerated performance on CPUs, GPUs, and TPUs.  
- Facilitates vectorization and parallelization for large-scale computations.  
- Uses a NumPy-like API, making it intuitive for scientific computing.  

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