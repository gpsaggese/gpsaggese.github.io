"""
Sklearn Model Trainer for TFX Pipeline

This module enables using sklearn models (XGBoost, RandomForest, etc.) in TFX pipelines
by wrapping them in TensorFlow SavedModel format for serving compatibility.
"""

import os
import pickle
import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict, Any
from tfx.components.trainer.fn_args_utils import FnArgs
import numpy as np

# Handle imports for both package and TFX wheel usage
try:
    from . import config
except ImportError:
    # When packaged as TFX wheel, define constants inline
    class config:
        TARGET_COLUMN = 'SalePrice'
        TF_DNN_PARAMS = {
            "batch_size": 32,
        }
# Import ModelRegistry only when needed (avoid circular imports in TFX wheel)


class SklearnModelWrapper(tf.Module):
    """
    Wrapper to convert sklearn model to TensorFlow SavedModel format.

    This allows sklearn models to be served using TensorFlow Serving.
    """

    def __init__(self, sklearn_model, transform_output):
        """
        Initialize the wrapper.

        Args:
            sklearn_model: Trained sklearn model (XGBoost, RandomForest, etc.)
            transform_output: TFX Transform component output path
        """
        super().__init__()
        self.sklearn_model = sklearn_model

        # Load TFTransform output and create trackable transform layer
        self.tf_transform_output = tft.TFTransformOutput(transform_output)
        self.tft_layer = self.tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve(self, serialized_examples):
        """
        Serving function that processes raw examples.

        Args:
            serialized_examples: Serialized tf.Example protos

        Returns:
            Predictions in original scale (house prices in dollars)
        """
        # Get feature spec and remove target (done inline to avoid tracking issues)
        feature_spec = self.tf_transform_output.raw_feature_spec()
        feature_spec.pop(config.TARGET_COLUMN, None)

        # Parse and transform the examples using the trackable transform layer
        parsed_features = tf.io.parse_example(serialized_examples, feature_spec)
        transformed_features = self.tft_layer(parsed_features)

        # Extract feature array for sklearn model
        feature_names = sorted(transformed_features.keys())
        feature_names = [f for f in feature_names if not f.startswith('SalePrice')]

        # Flatten and concatenate features to ensure 2D shape (batch_size, n_features)
        feature_tensors = []
        for name in feature_names:
            feat = tf.cast(transformed_features[name], tf.float32)
            # Reshape to ensure consistent 2D: (batch_size, feature_dim)
            feat = tf.reshape(feat, [tf.shape(feat)[0], -1])
            feature_tensors.append(feat)
        features_array = tf.concat(feature_tensors, axis=1)

        # Run sklearn model prediction using tf.py_function
        predictions = tf.py_function(
            func=self._predict_sklearn,
            inp=[features_array],
            Tout=tf.float32
        )

        # Reshape predictions
        predictions = tf.reshape(predictions, [-1, 1])

        # Convert from log space to original scale
        # exp(log(price + 1)) - 1 = price
        predictions = tf.exp(predictions) - 1.0

        return predictions

    def _predict_sklearn(self, features):
        """
        Make predictions using sklearn model.

        This runs in eager mode via tf.py_function.
        """
        # Convert to numpy
        features_np = features.numpy()

        # Predict using sklearn model
        predictions = self.sklearn_model.predict(features_np)

        return predictions.astype(np.float32)


def train_sklearn_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray
) -> Any:
    """
    Train a sklearn model.

    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training target (log-transformed)
        X_eval: Evaluation features
        y_eval: Evaluation target (log-transformed)

    Returns:
        Trained sklearn model
    """
    # Import here to avoid circular imports when TFX packages the module
    # Try relative import first, fall back to absolute import for TFX wheel
    try:
        from .model_comparison import ModelRegistry
    except ImportError:
        # When packaged as TFX wheel, use absolute import
        try:
            from model_comparison import ModelRegistry
        except ImportError:
            # If that also fails, create models inline
            import xgboost as xgb
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor,
                VotingRegressor, StackingRegressor
            )
            from sklearn.linear_model import Ridge, Lasso, ElasticNet

            # Define a minimal ModelRegistry for TFX wheel
            class ModelRegistry:
                @staticmethod
                def get_all_models():
                    # Base models
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=1000, max_depth=7, learning_rate=0.01,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        gamma=0, reg_alpha=0.1, reg_lambda=1, random_state=42,
                        objective='reg:squarederror', n_jobs=-1
                    )
                    rf_model = RandomForestRegressor(
                        n_estimators=500, max_depth=15, min_samples_split=5,
                        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
                    )
                    gbm_model = GradientBoostingRegressor(
                        n_estimators=1000, learning_rate=0.01, max_depth=5,
                        min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42
                    )
                    ridge_model = Ridge(alpha=10.0, random_state=42)

                    # Ensemble models need fresh instances
                    voting_ensemble = VotingRegressor(
                        estimators=[
                            ('xgb', xgb.XGBRegressor(
                                n_estimators=1000, max_depth=7, learning_rate=0.01,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                objective='reg:squarederror', n_jobs=-1
                            )),
                            ('rf', RandomForestRegressor(
                                n_estimators=500, max_depth=15, random_state=42, n_jobs=-1
                            )),
                            ('gbm', GradientBoostingRegressor(
                                n_estimators=1000, learning_rate=0.01, max_depth=5, random_state=42
                            ))
                        ],
                        n_jobs=-1
                    )

                    stacking_ensemble = StackingRegressor(
                        estimators=[
                            ('xgb', xgb.XGBRegressor(
                                n_estimators=1000, max_depth=7, learning_rate=0.01,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                objective='reg:squarederror', n_jobs=-1
                            )),
                            ('rf', RandomForestRegressor(
                                n_estimators=500, max_depth=15, random_state=42, n_jobs=-1
                            )),
                            ('gbm', GradientBoostingRegressor(
                                n_estimators=1000, learning_rate=0.01, max_depth=5, random_state=42
                            )),
                            ('ridge', Ridge(alpha=10.0, random_state=42))
                        ],
                        final_estimator=Ridge(alpha=1.0),
                        n_jobs=-1
                    )

                    return {
                        'XGBoost': xgb_model,
                        'RandomForest': rf_model,
                        'GradientBoosting': gbm_model,
                        'Ridge': ridge_model,
                        'Lasso': Lasso(alpha=0.001, max_iter=10000, random_state=42),
                        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42),
                        'VotingEnsemble': voting_ensemble,
                        'StackingEnsemble': stacking_ensemble
                    }

    import time

    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")

    # Get the model from registry
    print(f"Loading model from registry...", flush=True)
    models = ModelRegistry.get_all_models()

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    model = models[model_name]
    print(f"✓ Model loaded: {type(model).__name__}")

    # Train the model
    print(f"\nTraining on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"This may take 1-5 minutes depending on the model...")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.1f} seconds")

    # Evaluate
    print(f"\nEvaluating model performance...")
    train_score = model.score(X_train, y_train)
    eval_score = model.score(X_eval, y_eval)

    print(f"\n{'='*60}")
    print(f"TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Training R² score:   {train_score:.4f}")
    print(f"Validation R² score: {eval_score:.4f}")
    print(f"Training time:       {training_time:.1f}s")
    print(f"{'='*60}\n")

    return model


def run_fn(fn_args: FnArgs):
    """
    TFX Trainer run_fn for sklearn models.

    This function is called by the TFX Trainer component to train sklearn models.

    Args:
        fn_args: FnArgs object containing paths and parameters
    """
    print("\n" + "=" * 80)
    print("SKLEARN MODEL TRAINER FOR TFX")
    print("=" * 80)

    # Get model name from custom config (default to XGBoost)
    model_name = fn_args.custom_config.get('model_name', 'XGBoost') if fn_args.custom_config else 'XGBoost'
    print(f"\nSelected Model: {model_name}")
    print(f"Batch Size: {config.TF_DNN_PARAMS['batch_size']}")

    # Load transform output
    print(f"\n[STEP 1/6] Loading TFX Transform output...")
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    print(f"✓ Transform output loaded")

    # Load transformed training data
    print(f"\n[STEP 2/6] Loading transformed training data...")
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=config.TF_DNN_PARAMS['batch_size']
    )
    print(f"✓ Training dataset loaded")

    # Load transformed evaluation data
    print(f"\n[STEP 3/6] Loading transformed evaluation data...")
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=config.TF_DNN_PARAMS['batch_size']
    )
    print(f"✓ Evaluation dataset loaded")

    # Convert TF datasets to numpy arrays
    print(f"\n[STEP 4/6] Converting TensorFlow datasets to numpy arrays...")
    print("  Training dataset:")
    X_train, y_train = _dataset_to_numpy(train_dataset)
    print("  Evaluation dataset:")
    X_eval, y_eval = _dataset_to_numpy(eval_dataset)

    print(f"\n✓ Data conversion complete")
    print(f"  Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"  Evaluation: X={X_eval.shape}, y={y_eval.shape}")

    # Train the sklearn model
    print(f"\n[STEP 5/6] Training {model_name} model...")
    sklearn_model = train_sklearn_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval
    )

    # Wrap the sklearn model for TensorFlow Serving
    print(f"\n[STEP 6/6] Wrapping and saving model...")
    print("  Wrapping sklearn model in TensorFlow SavedModel format...")
    model_wrapper = SklearnModelWrapper(
        sklearn_model=sklearn_model,
        transform_output=fn_args.transform_output
    )

    # Define serving signature
    print("  Creating serving signature...")
    signatures = {
        'serving_default': model_wrapper.serve.get_concrete_function()
    }

    # Save the model
    print(f"  Saving to {fn_args.serving_model_dir}...")
    tf.saved_model.save(
        model_wrapper,
        fn_args.serving_model_dir,
        signatures=signatures
    )
    print(f"  ✓ TensorFlow SavedModel saved")

    # Also save the sklearn model as pickle for backup/inspection
    sklearn_model_path = os.path.join(fn_args.serving_model_dir, 'sklearn_model.pkl')
    with open(sklearn_model_path, 'wb') as f:
        pickle.dump(sklearn_model, f)
    print(f"  ✓ Sklearn pickle model saved")

    print("\n" + "=" * 80)
    print("✓ SKLEARN MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Location: {fn_args.serving_model_dir}")
    print("=" * 80 + "\n")


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=32):
    """
    Load and parse transformed data.

    Args:
        file_pattern: File pattern for transformed data (list of paths)
        data_accessor: Data accessor (not used, kept for compatibility)
        tf_transform_output: Transform output for parsing
        batch_size: Batch size

    Returns:
        TensorFlow dataset
    """
    # Get the feature spec for parsing
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Expand glob patterns to actual file paths
    file_list = []
    for pattern in file_pattern:
        matched_files = tf.io.gfile.glob(pattern)
        if matched_files:
            file_list.extend(matched_files)

    if not file_list:
        raise ValueError(f"No files found matching patterns: {file_pattern}")

    # Read TFRecord files directly (same as model_utils.py)
    dataset = tf.data.TFRecordDataset(file_list, compression_type='GZIP')

    # Parse examples
    def _parse_example(serialized_example):
        """Parse TFRecord example."""
        return tf.io.parse_single_example(serialized_example, transform_feature_spec)

    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _dataset_to_numpy(dataset):
    """
    Convert TF dataset to numpy arrays.

    Args:
        dataset: TensorFlow dataset

    Returns:
        Tuple of (X, y) numpy arrays
    """
    X_list = []
    y_list = []

    print("Converting TensorFlow dataset to numpy arrays...")
    batch_count = 0

    for batch in dataset:
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...", flush=True)

        # Extract features (exclude target)
        feature_names = sorted([k for k in batch.keys() if not k.startswith('SalePrice')])

        # Flatten and concatenate features to ensure 2D shape (batch_size, n_features)
        feature_tensors = []
        for name in feature_names:
            feat = tf.cast(batch[name], tf.float32)
            # Reshape to ensure consistent 2D: (batch_size, feature_dim)
            feat = tf.reshape(feat, [tf.shape(feat)[0], -1])
            feature_tensors.append(feat)
        features = tf.concat(feature_tensors, axis=1)
        X_list.append(features.numpy())

        # Extract target
        target_key = 'SalePrice_log'  # Transformed target name
        if target_key in batch:
            y_list.append(batch[target_key].numpy())
        else:
            # Try to find any SalePrice variant
            target_keys = [k for k in batch.keys() if k.startswith('SalePrice')]
            if target_keys:
                y_list.append(batch[target_keys[0]].numpy())

    print(f"  Completed: {batch_count} batches processed")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"  Final shapes: X={X.shape}, y={y.shape}")
    return X, y


if __name__ == "__main__":
    print("Sklearn Trainer Module for TFX")
    print("This module enables using sklearn models in TFX pipelines")
