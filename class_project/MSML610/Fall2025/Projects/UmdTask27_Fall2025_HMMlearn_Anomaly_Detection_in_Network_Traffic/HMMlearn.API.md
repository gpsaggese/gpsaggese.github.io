# TO DO: 
- (Document the native programming interface &#40;classes, functions, configuration objects&#41; of your chosen tool or library.)
- (Describe the lightweight wrapper layer you have written on top of this native API.)

-----

HMMlearn is a Python library designed for Hidden Markov Models &#40;HMM&#41;, which are statistical models that assume an underlying process generating observable events is a Markov process with unobserved &#40;hidden&#41; states. It is particularly useful for sequential data analysis and time-series prediction. 

Technologies Used
HMMlearn

- Provides implementations of various HMM algorithms, including Gaussian HMMs and Multinomial HMMs.
- Supports training, evaluation, and prediction on sequences of data.
- Offers functionality for model fitting and state inference.


**Project Objective**: Build an anomaly detection system for network traffic data using a Hidden Markov Model. The aim is to identify unusual patterns that may indicate security threats or breaches.

**Dataset Suggestions**: 
- Use the "UNSW-NB15" dataset available on Kaggle.
- Alternatively, utilize the "CICIDS 2017" dataset, which is also available on Kaggle.
> [!NOTE] 
> These datasets are not time series or sequential data. Therefore, they are NOT suitable for HMMs. I show this in the first part of my project notebook. Then, when I make it obvious, I use a similar dataset that is appropriate for this task:CESNET-TimeSeries24: Time Series Dataset for Network Traffic Anomaly Detection and Forecasting (https://zenodo.org/records/13382427)

**Tasks**:
- Data Preparation:
    - Preprocess the network traffic data, handling missing values and normalizing features.
  
- Feature Engineering:
    - Create time-series features such as packet counts, bytes transferred, and connection durations.
  
- Model Training:
    - Implement a Gaussian HMM using HMMlearn to model normal network behavior.
  
- Anomaly Detection:
    - Use the trained model to detect anomalies in live or historical network traffic data.
  
- Evaluation:
    - Assess the performance of the anomaly detection system using precision, recall, and F1-score metrics.
  
- Visualization:
    - Visualize detected anomalies and normal traffic patterns using Seaborn or Matplotlib.

**Bonus Ideas (Optional)**: 
- For Project 3, implement a feedback loop that retrains the model based on new traffic data to improve accuracy over time.



