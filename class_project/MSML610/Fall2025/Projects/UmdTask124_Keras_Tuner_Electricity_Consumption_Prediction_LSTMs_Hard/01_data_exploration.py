import pandas as pd
import matplotlib.pyplot as plt
import os

# simple path
DATA_DIR = "data"

# list all files
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") or f.endswith(".parquet") or f.endswith(".paruqet")]
print("Found files:")
for f in files:
    print("-", f)

# pick one to start — e.g. system-level PJM file if available
target_file = "PJM_Load_hourly.csv" if "PJM_Load_hourly.csv" in files else files[0]
print(f"\nUsing file for initial exploration: {target_file}")

# load the file
file_path = os.path.join(DATA_DIR, target_file)
df = pd.read_csv(file_path)

print("\n✅ File loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# detect datetime column
datetime_col = None
for c in df.columns:
    if "date" in c.lower() or "time" in c.lower():
        datetime_col = c
        break

if datetime_col is None:
    print("\n⚠️ Could not automatically detect a datetime column. Please check column names manually.")
else:
    print(f"\nDetected datetime column: {datetime_col}")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    print("\nTime range:", df.index.min(), "→", df.index.max())
    print("Frequency (approx):", df.index.to_series().diff().mode()[0])

# look for numeric columns
num_cols = df.select_dtypes(include='number').columns
print("\nNumeric columns:", num_cols.tolist()[:5])

# basic stats
print("\nSummary stats:")
print(df[num_cols].describe().T.head())

# check missing values
missing = df[num_cols].isna().mean() * 100
print("\nMissing values (%):")
print(missing.head())

# quick plot (daily aggregation)
if len(num_cols) > 0:
    df[num_cols[0]].resample("D").mean().plot(figsize=(10,5))
    plt.title(f"Daily avg of {num_cols[0]}")
    plt.ylabel("Value")
    plt.show()

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use the same df you already loaded (Datetime index, PJM_Load_MW column)
print("Starting from shape:", df.shape)

# -------------------------------------------------------------------
# 1. Ensure datetime continuity
# -------------------------------------------------------------------
# Reindex to continuous hourly intervals
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H")
df = df.reindex(full_range)

print("After reindexing:", df.shape)
print("Missing timestamps filled:", df.isna().sum().sum())

# If any missing load values, forward fill (though yours had none)
df["PJM_Load_MW"].fillna(method="ffill", inplace=True)

# -------------------------------------------------------------------
# 2. Create time-based features
# -------------------------------------------------------------------
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek           # Monday=0
df["month"] = df.index.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["dayofyear"] = df.index.dayofyear

# -------------------------------------------------------------------
# 3. Rolling statistics (helps model smooth seasonal patterns)
# -------------------------------------------------------------------
df["rolling_24h_mean"] = df["PJM_Load_MW"].rolling(window=24, min_periods=1).mean()
df["rolling_7d_mean"] = df["PJM_Load_MW"].rolling(window=24*7, min_periods=1).mean()

# -------------------------------------------------------------------
# 4. Optional: normalize features (for neural networks)
# -------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[["PJM_Load_MW","rolling_24h_mean","rolling_7d_mean"]])
df[["load_scaled","roll24_scaled","roll7_scaled"]] = scaled_features

# -------------------------------------------------------------------
# 5. Quick sanity checks
# -------------------------------------------------------------------
print("\nColumns after feature engineering:")
print(df.columns)

print("\nSample:")
print(df.head(3))

# -------------------------------------------------------------------
# 6. Correlation heatmap
# -------------------------------------------------------------------
import seaborn as sns
plt.figure(figsize=(8,5))
sns.heatmap(df[["PJM_Load_MW","hour","dayofweek","month","is_weekend","rolling_24h_mean","rolling_7d_mean"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations with Load")
plt.show()

# -------------------------------------------------------------------
# 7. Plot example trends
# -------------------------------------------------------------------
plt.figure(figsize=(12,5))
df["PJM_Load_MW"].resample("W").mean().plot(label="Weekly mean load")
df["rolling_7d_mean"].resample("W").mean().plot(label="7-day rolling mean", alpha=0.8)
plt.title("Electricity Load Trends (weekly)")
plt.legend()
plt.show()

# -------------------------------------------------------------------
# 8. Save processed file (optional)
# -------------------------------------------------------------------
df.to_csv("data/pjm_processed_features.csv")
print("✅ Saved feature-engineered data to data/pjm_processed_features.csv")

# %%

# -------------------------------------------------------------
# Step 2: Prepare LSTM-ready sequences and chronological splits
# -------------------------------------------------------------
import numpy as np

# ---- Parameters ----
SEQ_LEN = 24          # how many past hours to look at
TARGET_STEP = 1       # predict 1 hour ahead
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---- Choose feature columns ----
final_features = [
    "load_scaled", "roll24_scaled", "roll7_scaled",
    "hour_norm", "dayofweek_norm", "month_norm", "is_weekend"
]

# Normalize time-based features to 0-1 range
df["hour_norm"] = df["hour"] / 23.0
df["dayofweek_norm"] = df["dayofweek"] / 6.0
df["month_norm"] = (df["month"] - 1) / 11.0
df["is_weekend"] = df["is_weekend"].astype(float)

# ---- Prepare arrays ----
data_values = df[final_features].values
target_values = df["load_scaled"].values  # only the load as target

def create_sequences_single_target(features, target, seq_len=24, target_step=1):
    X, y = [], []
    for i in range(len(features) - seq_len - target_step + 1):
        X.append(features[i:i + seq_len])
        y.append(target[i + seq_len + target_step - 1])
    return np.array(X), np.array(y)

# ---- Build sequences ----
X, y = create_sequences_single_target(
    data_values, target_values,
    seq_len=SEQ_LEN, target_step=TARGET_STEP
)

print(f"✅ Sequence arrays created — X shape: {X.shape}, y shape: {y.shape}")
print("Each X sample shape:", X[0].shape)
print("Example target value:", y[0])

# ---- Chronological train/val/test split ----
n = len(X)
train_end = int(TRAIN_FRAC * n)
val_end = train_end + int(VAL_FRAC * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:], y[val_end:]

print("\n✅ Time-based splits:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---- Optional sanity check ----
print(f"\nTrain starts at index 0, ends at {train_end}")
print(f"Val starts at {train_end}, ends at {val_end}")
print(f"Test starts at {val_end}, ends at {n}")

# ---- Save ready arrays (optional) ----
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_val.npy", X_val)
np.save("data/y_val.npy", y_val)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("\n💾 Saved all prepared arrays inside the data/ folder")

# %%

# -------------------------------------------------------------
# Step 3: Baseline LSTM Model
# -------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# ---- Model Definition ----
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # output: next-hour load
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ---- Training ----
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ---- Evaluate ----
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Validation MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")

# ---- Plot Training Curves ----
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

# ---- Make Predictions ----
y_pred = model.predict(X_test)

# ---- Invert Scaling ----
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

# Reload scaler to get original MW scale (we already fit it earlier in feature engineering)
scaler = joblib.load("data/processed/scaler.pkl") if os.path.exists("data/processed/scaler.pkl") else None

# if scaler not saved, approximate inverse manually
y_pred_rescaled = y_pred
y_test_rescaled = y_test
if scaler:
    # scaler was fit on multiple columns, take the first (load) column min/max
    load_min = scaler.data_min_[0]
    load_max = scaler.data_max_[0]
    y_pred_rescaled = y_pred * (load_max - load_min) + load_min
    y_test_rescaled = y_test * (load_max - load_min) + load_min

# ---- Compute MAE/RMSE on original scale ----
mae = np.mean(np.abs(y_pred_rescaled - y_test_rescaled))
rmse = np.sqrt(np.mean((y_pred_rescaled - y_test_rescaled) ** 2))
print(f"\n📊 Baseline LSTM Performance on Test Set:")
print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")

# ---- Plot Actual vs Predicted ----
plt.figure(figsize=(10,5))
plt.plot(y_test_rescaled[:500], label='Actual')
plt.plot(y_pred_rescaled[:500], label='Predicted', alpha=0.7)
plt.title("Electricity Load Forecasting (First 500 Hours of Test Set)")
plt.xlabel("Time Steps (Hours)")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()

# %%

!pip install keras-tuner --quiet

# %%

# -------------------------------------------------------------
# Step 4: Hyperparameter Tuning with Keras Tuner
# -------------------------------------------------------------
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---- 1. Define model-building function ----
def build_model(hp):
    model = Sequential()
    
    # LSTM layer units
    hp_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_units,
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   return_sequences=False))
    
    # Dropout
    hp_dropout = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    model.add(Dropout(hp_dropout))
    
    # Dense layer size
    hp_dense = hp.Int('dense_units', min_value=16, max_value=128, step=16)
    model.add(Dense(units=hp_dense, activation='relu'))
    
    # Output
    model.add(Dense(1))
    
    # Learning rate
    hp_lr = hp.Choice('learning_rate', [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    
    model.compile(
        optimizer=Adam(learning_rate=hp_lr),
        loss='mse',
        metrics=['mae']
    )
    return model


# ---- 2. Initialize tuner (RandomSearch for speed) ----
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,            # increase to ~25+ for deeper search
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='pjm_lstm_tuning',
    overwrite=True
)

# ---- 3. Run the search ----
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

tuner.search(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ---- 4. Review best models ----
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n✅ Best Hyperparameters Found:")
print(f"LSTM units:   {best_hps.get('lstm_units')}")
print(f"Dense units:  {best_hps.get('dense_units')}")
print(f"Dropout rate: {best_hps.get('dropout_rate')}")
print(f"Learning rate:{best_hps.get('learning_rate')}")

# ---- 5. Build and retrain best model ----
best_model = tuner.hypermodel.build(best_hps)
history_best = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ---- 6. Evaluate on test set ----
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n🏁 Tuned Model Test MAE: {test_mae:.4f}")

# ---- 7. Plot training curves ----
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history_best.history['loss'], label='Train Loss')
plt.plot(history_best.history['val_loss'], label='Val Loss')
plt.title("Best Model - Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()

# %%

# -------------------------------------------------------------
# Step 5: Visualize Tuned Model Predictions vs Actual
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Predict on test set
y_pred = best_model.predict(X_test)

# ---- Invert scaling to MW (optional, if scaler saved earlier) ----
if os.path.exists("data/processed/scaler.pkl"):
    scaler = joblib.load("data/processed/scaler.pkl")
    load_min = scaler.data_min_[0]
    load_max = scaler.data_max_[0]
    y_pred_rescaled = y_pred * (load_max - load_min) + load_min
    y_test_rescaled = y_test * (load_max - load_min) + load_min
else:
    y_pred_rescaled = y_pred
    y_test_rescaled = y_test

# ---- Compute metrics ----
mae = np.mean(np.abs(y_pred_rescaled - y_test_rescaled))
rmse = np.sqrt(np.mean((y_pred_rescaled - y_test_rescaled) ** 2))
print(f"📈 Tuned LSTM — Test Performance:")
print(f"MAE  : {mae:.2f} MW")
print(f"RMSE : {rmse:.2f} MW")

# ---- Plot Actual vs Predicted ----
plt.figure(figsize=(12,5))
plt.plot(y_test_rescaled[:500], label="Actual Load")
plt.plot(y_pred_rescaled[:500], label="Predicted Load", alpha=0.8)
plt.title("PJM Hourly Load Forecast (Tuned LSTM, First 500 Hours of Test Set)")
plt.xlabel("Hour Index in Test Set")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()