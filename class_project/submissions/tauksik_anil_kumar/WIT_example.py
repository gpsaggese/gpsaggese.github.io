# WIT_example.py
# Credit Card Fraud Detection using Isolation Forest and What-If Tool
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Loading the dataset
data = pd.read_csv("creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Isolation Forest model
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_train)

# Predicting and mapping the results
y_pred = model.predict(X_test)
y_pred = [0 if p == 1 else 1 for p in y_pred]

# Evaluating
print("\n=== Model Evaluation ===")
print(classification_report(y_test, y_pred))

# Preparing the What-If Tool visualization
examples = X_test.sample(100, random_state=42)
builder = WitConfigBuilder(examples.to_dict(orient="records"), None)
WitWidget(builder)

