**Description**

ONNX (Open Neural Network Exchange) is an open-source format that enables the interoperability of machine learning models across various frameworks. It allows developers to transfer models between different platforms without the need for re-engineering. 

Technologies Used
ONNX

- Facilitates model conversion between popular frameworks like PyTorch, TensorFlow, and Scikit-learn.
- Supports a wide range of pre-trained models for various tasks.
- Optimizes models for performance on different hardware platforms.

---

## **Project 1: Classifying Fashion Items**  
**Difficulty**: 1 (Easy)  

**Project Objective**: Build an image classifier for clothing categories and convert the model into ONNX for cross-framework inference.  

**Dataset Suggestions**:  
- **Fashion-MNIST** — [Kaggle: Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  

**Tasks**:  
- Preprocess images (normalize, train/test split).  
- Train a **PyTorch CNN** to classify clothing items (t-shirt, dress, shoes, bag, etc.).  
- Convert the trained model to **ONNX**.  
- Run inference using **ONNX Runtime** or in TensorFlow.  
- Compare speed and accuracy between PyTorch and ONNX inference.  

**Bonus Ideas**:  
- Try a **MobileNetV2 pretrained model** and export to ONNX.  
- Apply **quantization** to reduce model size and test inference on CPU.  

---

## **Project 2: Fake News Detection**  
**Difficulty**: 2 (Medium)  

**Project Objective**: Build a text classification model to distinguish between real and fake news articles, then convert it to ONNX for deployment.  

**Dataset Suggestions**:  
- **Fake and Real News Dataset** — [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  

**Tasks**:  
- Preprocess text (cleaning, tokenization, padding).  
- Train a **TensorFlow/Keras LSTM or GRU model** for binary classification.  
- Convert the trained model to **ONNX**.  
- Perform inference with ONNX Runtime or load in PyTorch for evaluation.  
- Evaluate results using **accuracy, precision, recall, F1-score**.  

**Bonus Ideas**:  
- Fine-tune a **DistilBERT model** from HuggingFace and convert it to ONNX.  
- Deploy the ONNX model in a simple **FastAPI service** to detect fake news in real time.  

---

## **Project 3: Time Series Forecasting of Stock Prices**  
**Difficulty**: 3 (Hard)  

**Project Objective**: Build a time series forecasting model for stock prices using historical financial data, and use ONNX for cross-framework deployment.  

**Dataset Suggestions**:  
- **Price and Volume Data for All US Stocks & ETFs** — [Kaggle: Stock Prices Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)  

**Tasks**:  
- Preprocess stock price data (moving averages, volatility indicators).  
- Train a **TensorFlow/Keras LSTM model** for price forecasting.  
- Convert the model to **ONNX**.  
- Perform inference with ONNX Runtime and compare results across frameworks.  
- Evaluate forecasts using **MAE, RMSE, MAPE**.  

**Bonus Ideas**:  
- Compare ONNX-based deployment of **LSTM vs. Transformer** time-series models.  
- Build an **ensemble of ONNX models** for forecasting.  
- Create a **Streamlit dashboard** to visualize live forecasts using ONNX Runtime.  
