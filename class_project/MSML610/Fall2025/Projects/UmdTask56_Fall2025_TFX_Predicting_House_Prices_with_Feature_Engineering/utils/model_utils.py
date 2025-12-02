"""
Model training utilities for the house price prediction pipeline

This module contains:
- XGBoost model training functions
- TensorFlow DNN model training functions
- Model persistence and loading
- TFX Trainer run_fn for pipeline integration
"""

import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs

# Handle imports for both package and TFX wheel usage
try:
    from . import config
except ImportError:
    # When packaged as TFX wheel, define constants inline
    class config:
        TARGET_COLUMN = 'SalePrice'
        XGBOOST_PARAMS = {
            "n_estimators": 1000,
            "max_depth": 7,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "random_state": 42,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "early_stopping_rounds": 50
        }
        TF_DNN_PARAMS = {
            "hidden_units": [128, 64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2,
            "early_stopping_patience": 10
        }

# Import xgboost only when needed (not required for TFX Trainer)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def create_xgboost_model(params: Dict[str, Any] = None):
    """
    Create and configure XGBoost model.

    Args:
        params: XGBoost hyperparameters (optional)

    Returns:
        Configured XGBoost model

    TODO: Implement in Phase 4
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost")

    if params is None:
        params = config.XGBOOST_PARAMS

    model = xgb.XGBRegressor(**params)
    return model


def create_tensorflow_model(input_shape: int, params: Dict[str, Any] = None):
    """
    Create and configure TensorFlow DNN model.

    Args:
        input_shape: Number of input features
        params: Model hyperparameters (optional)

    Returns:
        Compiled Keras model

    TODO: Implement in Phase 4
    """
    if params is None:
        params = config.TF_DNN_PARAMS

    # Placeholder model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train XGBoost model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model parameters (optional)

    Returns:
        Trained model

    TODO: Implement in Phase 4
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost")

    model = create_xgboost_model(params)

    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
    else:
        model.fit(X_train, y_train)

    return model


def train_tensorflow_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train TensorFlow DNN model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model parameters (optional)

    Returns:
        Trained model

    TODO: Implement in Phase 4
    """
    if params is None:
        params = config.TF_DNN_PARAMS

    model = create_tensorflow_model(X_train.shape[1], params)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('early_stopping_patience', 10),
            restore_best_weights=True
        )
    ]

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

    return model, history


def test_model_training():
    """Test function to verify model utilities work."""
    print("Model training placeholder - will be implemented in Phase 4")


# ============================================================================
# TFX TRAINER FUNCTIONS
# ============================================================================

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a function that parses a serialized tf.Example and applies TFT.

    This function is used for serving the model.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(config.TARGET_COLUMN, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        # Convert dict of features to single tensor (same as training)
        features_list = [tf.cast(transformed_features[key], tf.float32)
                        for key in sorted(transformed_features.keys())]
        concatenated_features = tf.concat([tf.reshape(f, [tf.shape(f)[0], -1])
                                          for f in features_list], axis=1)

        # Get predictions in log space
        log_predictions = model(concatenated_features)

        # Transform back to original scale: exp(log(price + 1)) - 1 = price
        predictions = tf.exp(log_predictions) - 1.0

        return predictions

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[str],
              data_accessor,
              tf_transform_output,
              batch_size: int = 32) -> tf.data.Dataset:
    """
    Generates features and label for training/eval.

    Args:
        file_pattern: List of paths or patterns to TFRecord files
        data_accessor: DataAccessor for reading data
        tf_transform_output: TFT output wrapper
        batch_size: Batch size for training

    Returns:
        Dataset of (features, labels) tuples
    """
    # Get the feature spec for parsing
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Create a function to parse and split features/labels
    def _parse_and_split(serialized_example):
        """Parse TFRecord example and split into features and label."""
        parsed = tf.io.parse_single_example(serialized_example, transform_feature_spec)

        # Extract label
        label = parsed.pop('SalePrice_log')

        # Convert remaining features dict to a single tensor
        # Stack all feature values into a single vector
        features_list = [tf.cast(parsed[key], tf.float32) for key in sorted(parsed.keys())]
        features = tf.concat([tf.reshape(f, [-1]) for f in features_list], axis=0)

        return features, label

    # Expand glob patterns to actual file paths
    file_list = []
    for pattern in file_pattern:
        matched_files = tf.io.gfile.glob(pattern)
        if matched_files:
            file_list.extend(matched_files)

    if not file_list:
        raise ValueError(f"No files found matching patterns: {file_pattern}")

    # Read TFRecord files directly
    dataset = tf.data.TFRecordDataset(file_list, compression_type='GZIP')

    dataset = dataset.map(_parse_and_split, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _build_keras_model(input_dim: int,
                       hidden_units: List[int] = None,
                       learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Creates a DNN Keras model for house price prediction.

    Args:
        input_dim: Number of input features
        hidden_units: List of hidden layer sizes
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    if hidden_units is None:
        hidden_units = config.TF_DNN_PARAMS['hidden_units']

    # Input layer - receives concatenated transformed features
    inputs = tf.keras.Input(shape=(input_dim,), name='feature_input')

    # Build hidden layers
    x = inputs
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units,
            activation='relu',
            name=f'dense_{i}'
        )(x)
        x = tf.keras.layers.Dropout(0.2, name=f'dropout_{i}')(x)

    # Output layer for regression
    outputs = tf.keras.layers.Dense(1, name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )

    return model


def run_fn(fn_args: FnArgs):
    """
    Train the model based on given args.

    This is the main entry point for TFX Trainer component.

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    print("\n" + "=" * 80)
    print("TENSORFLOW DNN TRAINER FOR TFX")
    print("=" * 80)

    print(f"\nModel Architecture: Deep Neural Network")
    print(f"Hidden Units: {config.TF_DNN_PARAMS['hidden_units']}")
    print(f"Batch Size: {config.TF_DNN_PARAMS['batch_size']}")
    print(f"Max Epochs: {config.TF_DNN_PARAMS['epochs']}")
    print(f"Early Stopping Patience: {config.TF_DNN_PARAMS.get('early_stopping_patience', 10)}")

    # Load the transform output
    print(f"\n[STEP 1/6] Loading TFX Transform output...")
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    print(f"✓ Transform output loaded")

    # Get training and eval datasets
    print(f"\n[STEP 2/6] Loading transformed training data...")
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=config.TF_DNN_PARAMS['batch_size']
    )
    print(f"✓ Training dataset loaded")

    print(f"\n[STEP 3/6] Loading transformed evaluation data...")
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=config.TF_DNN_PARAMS['batch_size']
    )
    print(f"✓ Evaluation dataset loaded")

    # Determine input dimension from transformed features
    # Count all features except the label
    print(f"\n[STEP 4/6] Building model architecture...")
    transform_feature_spec = tf_transform_output.transformed_feature_spec()
    feature_keys = [k for k in transform_feature_spec.keys() if k != 'SalePrice_log']
    input_dim = len(feature_keys)
    print(f"  Input dimension: {input_dim} features")

    # Build the model
    model = _build_keras_model(
        input_dim=input_dim,
        hidden_units=config.TF_DNN_PARAMS['hidden_units'],
        learning_rate=config.TF_DNN_PARAMS['learning_rate']
    )
    print(f"✓ Model built successfully")
    model.summary()

    # Set up callbacks
    print(f"\n[STEP 5/6] Training TensorFlow DNN model...")
    print(f"  This may take 2-5 minutes...")
    print(f"  You'll see epoch-by-epoch progress below:\n")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq='batch'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.TF_DNN_PARAMS.get('early_stopping_patience', 10),
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=config.TF_DNN_PARAMS['epochs'],
        callbacks=[tensorboard_callback, early_stopping],
        verbose=1
    )

    print(f"\n✓ Training completed")
    print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # Create a new model with the serving signature
    # This wraps the model to handle raw tf.Examples
    print(f"\n[STEP 6/6] Creating serving signature and saving model...")
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples'
                )
            ),
    }

    # Save the model
    print(f"  Saving to {fn_args.serving_model_dir}...")
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
    print(f"  ✓ TensorFlow SavedModel saved")

    print("\n" + "=" * 80)
    print("✓ TENSORFLOW DNN TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModel: TensorFlow Deep Neural Network")
    print(f"Location: {fn_args.serving_model_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_model_training()
