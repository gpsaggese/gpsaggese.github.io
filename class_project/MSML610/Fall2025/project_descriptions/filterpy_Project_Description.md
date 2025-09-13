## Description  
FilterPy is a Python library that provides a collection of filtering and estimation algorithms, particularly for Kalman filters and related techniques. It is widely used for tasks involving state estimation, tracking, and sensor fusion. The library supports various types of filters, including linear and non-linear models, making it suitable for a range of applications in data science and machine learning.  

---

## Project 1: Simple Object Tracking Using Kalman Filters  
**Difficulty**: 1 (Easy)  

**Project Objective**  
Implement a basic Kalman filter to track the motion of a simple object (e.g., a ball) in 2D space and compare with baseline ML models.  

**Dataset Suggestions**  
- Synthetic data: simulate 2D straight-line motion with Gaussian noise.  
- Real data option: [Basketball Object Tracking Dataset (Kaggle)](https://www.kaggle.com/datasets/trainingdatapro/basketball-tracking-dataset) — ball tracking during basketball gameplay.  

**Tasks**  
- **Simulate or Load Data**: Create noisy motion data or use the basketball dataset.  
- **Kalman Filter**: Apply FilterPy’s Kalman filter for position estimation.  
- **ML Model Comparisons**:  
  - **Linear Regression** for trajectory fitting.  
  - **ARIMA** for time-series forecasting.  
- **Visualization**: Plot actual vs. predicted trajectories.  
- **Evaluation**: Calculate Mean Squared Error (MSE) for comparison.  

**Bonus Ideas (Optional)**  
- Extend to multiple object tracking.  
- Add acceleration/noise for non-linear motion and use an Extended Kalman Filter (EKF).  

---

## Project 2: Sensor Fusion for Environmental Monitoring  
**Difficulty**: 2 (Medium)  

**Project Objective**  
Fuse data from multiple environmental sensors (temperature, CO, NO₂, etc.) using Kalman filters, and compare with ML models.  

**Dataset Suggestions**  
[Air Quality Data Set (UCI)](https://archive.ics.uci.edu/ml/datasets/air+quality) — 9,358 hourly readings of pollutants (CO, NO₂, O₃) and meteorological variables.  

**Tasks**  
- **Data Preprocessing**: Clean missing values (coded as “−200”), resample if needed.  
- **Kalman Filter**: Implement multi-sensor fusion with FilterPy.  
- **ML Model Comparisons**:  
  - **Random Forest Regressor** for supervised prediction.  
  - **XGBoost Regressor** for robust handling of noise.  
- **Evaluation**: Compare fused Kalman estimates vs ML predictions using MAE and RMSE.  
- **Visualization**: Plot pollutant levels with fused vs raw sensor values.  

**Bonus Ideas (Optional)**  
- Test robustness with artificially added noise.  
- Use an Extended Kalman Filter (EKF) for non-linear pollutant interactions.  

---

## Project 3: Autonomous Vehicle Navigation with EKF  
**Difficulty**: 3 (Hard)  

**Project Objective**  
Develop an Extended Kalman Filter (EKF) for vehicle localization by combining GPS and IMU data, and benchmark against ML approaches.  

**Dataset Suggestions**  
- [NCLT Dataset (University of Michigan)](http://robots.engin.umich.edu/nclt/) — includes GPS and IMU sensor data (manageable subsets available).  
- Alternative: [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/) (large; students should use GPS+IMU subset only).  

**Tasks**  
- **Data Loading**: Preprocess GPS and IMU sensor data.  
- **Extended Kalman Filter**: Apply EKF with FilterPy for localization.  
- **ML Model Comparisons**:  
  - **LSTM (Keras)** for sequence-based trajectory prediction.  
  - **XGBoost Regressor** as a tabular baseline.  
- **Evaluation**: Compare EKF vs ML predictions using RMSE of predicted vs ground truth trajectories.  
- **Visualization**: Plot estimated vs true paths on a 2D map.  

**Bonus Ideas (Optional)**  
- Add LiDAR or camera features for multi-sensor fusion.  
- Stress-test models under sensor dropout or increased noise.  

---
