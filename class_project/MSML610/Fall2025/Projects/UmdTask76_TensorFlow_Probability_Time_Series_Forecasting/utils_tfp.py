import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from statsmodels.tsa.arima.model import ARIMA
import os

# ==========================================
# 1. Data Processing Utils
# ==========================================
def load_and_clean_air_quality_data(csv_path):
    """
    Loads UCI Air Quality data, parses datetimes, and enforces hourly frequency.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find data file at: {csv_path}")

    raw = pd.read_csv(csv_path, sep=';', decimal=',')
    
    # Parse DateTime
    df = raw.copy()
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').set_index('Datetime')
    
    # Handle Missing Values (-200 -> NaN)
    target_col = 'CO(GT)'
    series = df[target_col].replace(-200, np.nan)
    
    # Enforce Hourly Frequency (Interpolate gaps)
    series = series.resample('h').asfreq()
    series = series.interpolate(method='linear')
    
    return series

def split_train_test(series, split_ratio=0.8):
    """Splits the series into train and test sets."""
    train_size = int(len(series) * split_ratio)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    return train, test

# ==========================================
# 2. Gaussian Process Model Utils
# ==========================================
class TFP_GaussianProcess_Forecaster:
    """
    A wrapper class for TensorFlow Probability Gaussian Processes.
    """
    def __init__(self, train_values, train_indices=None):
        self.y_train = train_values.reshape(-1, 1).astype(np.float64)
        if train_indices is None:
            self.X_train = np.arange(len(train_values)).reshape(-1, 1).astype(np.float64)
        else:
            self.X_train = train_indices.reshape(-1, 1).astype(np.float64)
            
        # Initialize Variables
        self.amplitude = tf.Variable(1.0, dtype=tf.float64, name='amplitude')
        self.length_scale = tf.Variable(10.0, dtype=tf.float64, name='length_scale')
        self.observation_noise_variance = tf.Variable(0.1, dtype=tf.float64, name='noise_var')
        
        # Build Kernel (RBF + Periodic)
        kernel_rbf = tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=self.amplitude, 
            length_scale=self.length_scale
        )
        kernel_periodic = tfp.math.psd_kernels.ExpSinSquared(
            amplitude=tf.Variable(1.0, dtype=tf.float64),
            length_scale=tf.Variable(1.0, dtype=tf.float64),
            period=tf.Variable(24.0, dtype=tf.float64)
        )
        self.kernel = kernel_rbf + kernel_periodic
        
    def train(self, epochs=100, learning_rate=0.05):
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                gp = tfd.GaussianProcess(
                    kernel=self.kernel,
                    index_points=self.X_train,
                    observation_noise_variance=self.observation_noise_variance
                )
                loss = -gp.log_prob(self.y_train[:, 0])
            grads = tape.gradient(loss, tape.watched_variables())
            optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            return loss

        print(f"Training GP for {epochs} epochs...")
        for i in range(epochs):
            loss = train_step()
            if i % 50 == 0:
                print(f"Epoch {i}: Loss = {loss.numpy():.4f}")
                
    def predict(self, horizon):
        X_test = np.arange(len(self.y_train), len(self.y_train) + horizon).reshape(-1, 1).astype(np.float64)
        
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=X_test,
            observation_index_points=self.X_train,
            observations=self.y_train[:, 0],
            observation_noise_variance=self.observation_noise_variance
        )
        return gprm.mean().numpy(), gprm.stddev().numpy()

# ==========================================
# 3. ARIMA Baseline Utils
# ==========================================
class Baseline_ARIMA_Forecaster:
    """
    A wrapper for the Statsmodels ARIMA model.
    """
    def __init__(self, train_data, order=(24, 1, 0)):
        self.train_data = train_data
        self.order = order
        self.model_fit = None
        
    def train(self):
        print(f"Training ARIMA model with order {self.order}...")
        model = ARIMA(self.train_data, order=self.order)
        self.model_fit = model.fit()
        print("ARIMA training complete.")
        
    def predict(self, steps):
        if self.model_fit is None:
            raise ValueError("Model not trained yet!")
        forecast_res = self.model_fit.get_forecast(steps=steps)
        return forecast_res.predicted_mean, forecast_res.conf_int()