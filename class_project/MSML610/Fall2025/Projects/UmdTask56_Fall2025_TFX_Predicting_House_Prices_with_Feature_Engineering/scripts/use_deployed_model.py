"""
Script to use the deployed model for predictions.

This script shows three ways to use the deployed StackingEnsemble model:
1. Load the TensorFlow SavedModel
2. Load the sklearn pickle directly
3. Make predictions on the test set
"""

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf


def load_tensorflow_savedmodel(model_path):
    """Load the TensorFlow SavedModel format."""
    print(f"\n{'='*60}")
    print("Loading TensorFlow SavedModel")
    print(f"{'='*60}")

    model = tf.saved_model.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    print(f"Available signatures: {list(model.signatures.keys())}")

    return model


def load_sklearn_model(model_path):
    """Load the sklearn model from pickle."""
    print(f"\n{'='*60}")
    print("Loading sklearn pickle model")
    print(f"{'='*60}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"✓ Model loaded from {model_path}")
    print(f"Model type: {type(model).__name__}")

    return model


def predict_with_savedmodel(model, test_csv_path):
    """
    Make predictions using the TensorFlow SavedModel.

    Note: This requires raw examples to be serialized as tf.Example protos.
    """
    print(f"\n{'='*60}")
    print("Making predictions with TensorFlow SavedModel")
    print(f"{'='*60}")

    # Load test data
    df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(df)} test examples")

    # Convert to tf.Example format (simplified version)
    # In production, you'd use the same preprocessing as training
    print("\nNote: TensorFlow SavedModel requires tf.Example format.")
    print("For production use, preprocess data through the Transform component.")
    print("For direct predictions, use the sklearn model (Method 2).")

    return None


def predict_with_sklearn(model, transformed_test_data):
    """
    Make predictions using the sklearn model.

    Args:
        model: Loaded sklearn model
        transformed_test_data: Preprocessed test features (numpy array)

    Returns:
        Predictions array
    """
    print(f"\n{'='*60}")
    print("Making predictions with sklearn model")
    print(f"{'='*60}")

    predictions = model.predict(transformed_test_data)

    # Convert from log space to original scale
    # exp(log(price + 1)) - 1 = price
    predictions = np.exp(predictions) - 1.0

    print(f"✓ Generated {len(predictions)} predictions")
    print(f"Prediction range: ${predictions.min():,.2f} - ${predictions.max():,.2f}")
    print(f"Mean prediction: ${predictions.mean():,.2f}")

    return predictions


def main():
    """Main function to demonstrate model usage."""

    # Find the latest model in serving directory
    serving_dir = "models/serving"
    model_versions = sorted([d for d in os.listdir(serving_dir) if d.isdigit()], reverse=True)

    if not model_versions:
        print("ERROR: No model versions found in models/serving/")
        return

    latest_version = model_versions[0]
    model_path = os.path.join(serving_dir, latest_version)

    print(f"\n{'='*80}")
    print("DEPLOYED MODEL USAGE GUIDE")
    print(f"{'='*80}")
    print(f"Latest model version: {latest_version}")
    print(f"Model path: {model_path}")

    # Method 1: Load TensorFlow SavedModel
    try:
        tf_model = load_tensorflow_savedmodel(model_path)
        print("\n✓ Method 1: TensorFlow SavedModel loaded successfully")
    except Exception as e:
        print(f"\n✗ Method 1 failed: {e}")

    # Method 2: Load sklearn pickle
    try:
        sklearn_model_path = os.path.join(model_path, "sklearn_model.pkl")
        sklearn_model = load_sklearn_model(sklearn_model_path)
        print("\n✓ Method 2: sklearn pickle loaded successfully")

        # Show model details
        print(f"\nModel Details:")
        print(f"  Final estimator: {type(sklearn_model.final_estimator_).__name__}")
        print(f"  Base estimators: {len(sklearn_model.estimators)} models")
        for name, estimator in sklearn_model.estimators:
            print(f"    - {name}: {type(estimator).__name__}")

    except Exception as e:
        print(f"\n✗ Method 2 failed: {e}")

    # Method 3: Example prediction workflow
    print(f"\n{'='*60}")
    print("PREDICTION WORKFLOW")
    print(f"{'='*60}")
    print("\nTo make predictions on new data:")
    print("1. Preprocess data using the Transform component")
    print("2. Load transformed features from pipeline outputs")
    print("3. Use sklearn model to predict")
    print("4. Transform predictions back to original scale")

    print("\nExample code:")
    print("""
    # Load transformed test data
    import tensorflow as tf
    import tensorflow_transform as tft

    # Load transform output
    transform_output = tft.TFTransformOutput('pipeline_outputs/.../Transform/transform_graph/...')

    # Load transformed data
    dataset = tf.data.TFRecordDataset(['pipeline_outputs/.../Transform/transformed_examples/...'])

    # Convert to numpy arrays (as in sklearn_trainer.py)
    X_test = # ... load and stack features

    # Load sklearn model and predict
    with open('models/serving/{}/sklearn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    predictions = np.exp(predictions) - 1.0  # Convert from log space
    """.format(latest_version))

    print(f"\n{'='*80}")
    print("For full pipeline predictions, use the Evaluator component or")
    print("create a custom prediction script that loads transformed data.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
