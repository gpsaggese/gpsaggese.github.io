## Airport Congestion Level Prediction using XGBoost

### Project Overview

This project predicts hourly airport congestion levels for major U.S. airports using real flight operation data and a machine-learning model based on XGBoost.
The goal is to provide an interpretable and interactive way to understand congestion patterns across flights and hours.
```text
The project includes:
	•	Data preprocessing and feature engineering
	•	Exploratory data analysis (EDA)
	•	Model training using XGBoost
	•	An interactive Streamlit web application for visualization and prediction
```

### File Descriptions

`API.ipynb`
```text
A notebook that demonstrates API-style usage of the trained model, including:
	•	Loading the saved XGBoost model
	•	Accepting structured input (airport, hour, flight attributes)
	•	Generating congestion predictions programmatically
	•	Showing how the model could be integrated into external systems
```
This notebook simulates how the model would behave if exposed via a backend service or REST API.

`API.md`
```text
A Markdown document that explains API.ipynb, including:
	•	Expected input schema
	•	Output format
	•	Example prediction calls
	•	Intended usage scenarios
```
This file is useful for developers, reviewers, and future system integration.

`example.ipynb`

```text
A single consolidated Jupyter Notebook that contains:
	•	Data loading and cleaning
	•	Exploratory Data Analysis (EDA)
	•	Feature engineering
	•	Model training and evaluation using XGBoost
```

All previously separate notebooks have been merged into this one file to provide a complete and reproducible workflow.

`example.md`

```text
A Markdown explanation of example.ipynb.
This file explains:
	•	The purpose of each major code section
	•	Why specific features and techniques were used
	•	How results should be interpreted
```
This file is useful for documentation, grading, and non-technical readers.

`util_preprocess_hourly.py`

```text
Handles data preprocessing and feature creation, including:
	•	Cleaning raw flight data
	•	Aggregating data at the hourly level
	•	Creating congestion-related features
	•	Saving the processed dataset for modeling
```

`util_train_model.py`

```text
Responsible for model training, including:
	•	Loading processed data
	•	Training the XGBoost model
	•	Evaluating performance
	•	Saving the trained model for later use
```

`util_streamlit_operations.py`
```text
Contains all Streamlit-related logic, including:
   •	Loading the trained model
   •	Handling user input (airport and hour selection)
   •	Generating visualizations (bar charts and tables)
   •	Displaying congestion predictions interactively
```
This file was renamed from app.py to better reflect its functional role.

data/
	•	raw/ – Original datasets (not modified)
	•	processed/ – Cleaned and feature-engineered data
	•	models/ – Saved trained model files

requirements.txt

Lists all Python dependencies required to run the project.

⸻
### How to Run the Project

1. Install Dependencies

pip install -r requirements.txt

streamlit run util_streamlit_operations.py

⸻

### Web Application Functionality
	•	Select an airport and hour
	•	View congestion predictions through:
	•	Interactive bar charts
	•	A summary table showing time, flight number, and congestion level
	•	Hover over bars to see:
	•	Airport
	•	Hour
	•	Predicted congestion level

⸻

### Key Technologies Used
	•	Python
	•	Pandas, NumPy
	•	XGBoost
	•	Scikit-learn
	•	Streamlit
	•	Plotly / Matplotlib


### Project Structure

All project files are organized directly inside the UmdTask folder.

```text
UmdTask/
│
├── example.ipynb
├── example.md
├── API.ipynb
├── API.md
├── util_preprocess_hourly.py
├── util_train_model.py
├── util_streamlit_operations.py
├── requirements.txt
├── README.md
└── data/
    ├── raw/
    ├── processed/
    └── models/

```

### Author

Varun Parashar

MSML610 — Fall 2025

University of Maryland
