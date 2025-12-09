import numpy as np
import os
import json
from typing import Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import onnxruntime as ort


def _convert_keras_to_onnx_without_cudnn(model_path: str, onnx_path: str) -> str:
    """
    Convert Keras model to ONNX by recreating it without CuDNN optimization. I wasnt able to convert to ONNX with CuDNN

    This is needed because CuDNN LSTM operations are not supported in ONNX.

    Args:
        model_path: Path to Keras model
        onnx_path: Path to save ONNX model

    Returns:
        Path to saved ONNX model
    """
    original_model = keras.models.load_model(model_path, compile=False)
    config = original_model.get_config()

    # Recreate model with use_cudnn=False for LSTM layers
    for layer_config in config['layers']:
        layer_class = layer_config['class_name']

        if layer_class == 'LSTM':
            # Add use_cudnn=False to LSTM layer config
            layer_config['config']['use_cudnn'] = False
        elif layer_class == 'Bidirectional':
            # Handle Bidirectional LSTM
            if layer_config['config']['layer']['class_name'] == 'LSTM':
                layer_config['config']['layer']['config']['use_cudnn'] = False

    # Create new model from modified config
    # Use the appropriate from_config method based on model type
    if isinstance(original_model, keras.Sequential):
        new_model = keras.Sequential.from_config(config)
    else:
        new_model = keras.Model.from_config(config)

    # Copy weights from original model to new model
    new_model.set_weights(original_model.get_weights())

    # Keras 3 requires the model to be called at least once before export
    input_shape = new_model.input_shape
    dummy_shape = tuple(1 if dim is None else dim for dim in input_shape)
    dummy_input = np.zeros(dummy_shape, dtype=np.float32)

    # Call the model to build it
    new_model(dummy_input)

    new_model.export(onnx_path, format="onnx")

    return onnx_path


def convert_to_onnx(model_path: str, onnx_path: str) -> str:
    """
    Convert Keras model to ONNX format.

    Args:
        model_path: Path to Keras model (.h5, .keras, or SavedModel)
        onnx_path: Path to save ONNX model

    Returns:
        Path to saved ONNX model
    """
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if model_path.endswith('.keras'):
        onnx_path = _convert_keras_to_onnx_without_cudnn(model_path, onnx_path)

    return onnx_path


def verify_onnx(onnx_path: str) -> Dict[str, Any]:
    """
    Verify ONNX model using onnx.checker.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Verification results
    """
    model = onnx.load(onnx_path)

    try:
        onnx.checker.check_model(model)
        is_valid = True
        error = None
    except Exception as e:
        is_valid = False
        error = str(e)

    return {
        'is_valid': is_valid,
        'error': error,
        'opset_version': model.opset_import[0].version,
        'num_nodes': len(model.graph.node)
    }


class ONNXInferenceSession:
    """Wrapper class for ONNX Runtime inference session."""

    def __init__(self, model_path: str):
        """
        Initialize ONNX Runtime session.

        Args:
            model_path: Path to ONNX model
        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX Runtime.

        Args:
            X: Input array

        Returns:
            Predictions
        """
        return self.session.run([self.output_name], {self.input_name: X.astype(np.float32)})[0]

    def get_input_shape(self):
        """Get expected input shape."""
        return self.session.get_inputs()[0].shape

    def get_output_shape(self):
        """Get output shape."""
        return self.session.get_outputs()[0].shape


def compare_frameworks_inference(
    keras_model_path: str,
    onnx_model_path: str,
    test_input: np.ndarray
) -> Dict[str, Any]:
    """
    Compare inference results between TensorFlow and ONNX Runtime.

    Args:
        keras_model_path: Path to Keras model
        onnx_model_path: Path to ONNX model
        test_input: Test input array

    Returns:
        Comparison results
    """
    import time

    # Load model without compiling to avoid deserialization issues
    keras_model = keras.models.load_model(keras_model_path, compile=False)

    start = time.time()
    tf_pred = keras_model.predict(test_input, verbose=0)
    tf_time = time.time() - start

    onnx_session = ONNXInferenceSession(onnx_model_path)

    start = time.time()
    onnx_pred = onnx_session.predict(test_input)
    onnx_time = time.time() - start

    max_diff = np.max(np.abs(tf_pred - onnx_pred))
    mean_diff = np.mean(np.abs(tf_pred - onnx_pred))

    return {
        'tensorflow_time': tf_time,
        'onnx_time': onnx_time,
        'speedup': tf_time / onnx_time,
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'numerically_close': np.allclose(tf_pred, onnx_pred, rtol=1e-4, atol=1e-4)
    }