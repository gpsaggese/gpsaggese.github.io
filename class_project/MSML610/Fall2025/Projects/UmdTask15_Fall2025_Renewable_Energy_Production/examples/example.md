# Example Notebook — Renewable Energy Forecasting

This document summarizes the purpose and workflow of the
`example.ipynb` notebook.  
The notebook demonstrates a complete, practical walkthrough of how to
use the project utilities to explore the dataset, engineer features,
train a baseline model, and evaluate forecasting performance.

The dataset consists of hourly solar energy production paired with weather variables (temperature, cloud cover, wind speed, solar radiation) over a multi-month period.

The goal of the example notebook is to show a **typical end-to-end
workflow** that a user or teammate could follow when working with the
forecasting system.

---

## What This Notebook Demonstrates

### **1. Dataset Exploration (EDA)**
The notebook begins with exploratory data analysis on the raw solar
dataset, including:

- Inspecting columns and data types  
- Understanding the time range and sampling frequency  
- Visualizing trends and distributions in solar production and weather  
- Checking for patterns such as daily cycles and variability  

This helps build intuition about the data before modeling.

---

### **2. Feature Engineering**
The example notebook uses the project’s feature-engineering utilities to
transform the dataset by adding:

- Time features (hour, day of week, month)  
- Lag features  
- Rolling window averages  

This results in a feature-rich dataset suitable for forecasting.

---

### **3. Time-Based Train/Validation Split**
The notebook shows how to create a proper **time-ordered split** using
the utility functions, assigning the most recent days to validation to
simulate real forecasting conditions.

This ensures the model is evaluated on future data rather than shuffled
samples.

---

### **4. Baseline Model Training**
A baseline model (Random Forest) is trained using the engineered
features and the time-based split.

The notebook demonstrates:

- Fitting the model  
- Obtaining validation predictions  
- Computing basic metrics such as MAE and RMSE  

This provides a simple, interpretable benchmark for the project.

---

### **5. Visualizing Forecast Performance**
The notebook includes a comparison of actual vs. predicted energy
production on the validation set.

This plot helps illustrate:

- How well the model follows real production patterns  
- Any areas of over- or under-prediction  
- Overall forecast shape and timing  

---

## Summary

The `example.ipynb` notebook provides a clear, reproducible example of:

1. Exploring the raw solar dataset  
2. Creating forecasting features  
3. Performing a correct time-based split  
4. Training a baseline model  
5. Evaluating and visualizing predictions  

It demonstrates how the project’s utilities and workflow come together
in practice, giving users and reviewers a complete overview of the
system’s functionality.
