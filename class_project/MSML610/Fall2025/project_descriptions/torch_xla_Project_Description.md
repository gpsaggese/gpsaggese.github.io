**Description**

In this project, students will utilize Torch XLA, a library that accelerates PyTorch models on TPUs (Tensor Processing Units) for enhanced performance. Torch XLA allows seamless integration with PyTorch, enabling faster training and inference of deep learning models. This tool is particularly useful for large-scale machine learning tasks, where computational efficiency is crucial.

Technologies Used
Torch XLA

- Accelerates PyTorch models on TPUs for improved performance.
- Offers seamless integration with existing PyTorch codebases.
- Provides utilities for distributed training and efficient data loading.

---

### Project 1: Image Classification with Transfer Learning
**Difficulty**: 1 (Easy)

**Project Objective**: Build a model to classify images from the CIFAR-10 dataset, optimizing for accuracy using transfer learning techniques.

**Dataset Suggestions**: 
- CIFAR-10 dataset available on Kaggle: [CIFAR-10](https://www.kaggle.com/c/cifar-10)

**Tasks**:
- Set Up Environment:
  - Install Torch XLA and configure it to use TPU.
  
- Load and Preprocess Data:
  - Import the CIFAR-10 dataset and perform basic preprocessing (normalization, resizing).
  
- Implement Transfer Learning:
  - Utilize a pre-trained model (e.g., ResNet) and modify it for CIFAR-10 classification.
  
- Train the Model:
  - Train the model on the TPU, monitoring loss and accuracy.
  
- Evaluate Model Performance:
  - Assess the model's accuracy on a test set and analyze misclassifications.
  
- Visualization:
  - Create visualizations of sample predictions and confusion matrix.

---

### Project 2: Natural Language Processing with Transformers
**Difficulty**: 2 (Medium)

**Project Objective**: Develop a sentiment analysis model using the IMDb movie reviews dataset, optimizing for F1 score through fine-tuning a transformer model.

**Dataset Suggestions**: 
- IMDb dataset available on Kaggle: [IMDb Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**Tasks**:
- Set Up Environment:
  - Configure Torch XLA for TPU usage and install necessary libraries (Transformers).

- Load and Preprocess Data:
  - Load the IMDb dataset and preprocess text (tokenization, padding).
  
- Model Selection:
  - Choose a pre-trained transformer model (e.g., BERT) for sentiment classification.
  
- Fine-Tuning:
  - Fine-tune the model on the IMDb dataset using TPU for accelerated training.
  
- Evaluate Model:
  - Evaluate the model’s performance using F1 score and confusion matrix.
  
- Visualization:
  - Visualize the distribution of sentiments and model predictions.

---

### Project 3: Time Series Forecasting with LSTM
**Difficulty**: 3 (Hard)

**Project Objective**: Create a deep learning model to forecast stock prices using historical data from the Yahoo Finance API, optimizing for Mean Absolute Error (MAE).

**Dataset Suggestions**: 
- Historical stock prices available via Yahoo Finance API: [Yahoo Finance API](https://www.yahoofinanceapi.com/)

**Tasks**:
- Set Up Environment:
  - Install and configure Torch XLA to work with TPUs.

- Data Ingestion:
  - Use the Yahoo Finance API to fetch historical stock price data for a selected company.
  
- Data Preprocessing:
  - Clean and preprocess the data (handling missing values, normalization).
  
- Build LSTM Model:
  - Construct an LSTM model architecture suitable for time series forecasting.
  
- Train the Model:
  - Train the model on TPU, optimizing for MAE and adjusting hyperparameters.
  
- Evaluate Model:
  - Assess the forecasting accuracy using MAE and visualize the predicted vs. actual prices.
  
- Bonus Ideas:
  - Experiment with different time windows for input data.
  - Implement additional features such as moving averages or technical indicators.
  - Compare LSTM performance with other models (e.g., ARIMA, GRU).

