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


def convert_tcn_to_onnx(
    model,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 12
) -> str:
    """
    Convert DARTS TCN model to ONNX format.

    Args:
        model: Trained DARTS TCNModel
        input_shape: Input shape (sequence_length, n_features)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    import torch
    import torch.nn as nn

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Get the underlying PyTorch model and move to CPU for ONNX export
        torch_model = model.model
        torch_model.cpu()  # Move model to CPU to avoid device mismatch
        torch_model.float()  # Convert to Float32 to match dummy input dtype
        torch_model.eval()

        # DARTS TCN models with past_covariates expect input as (target + past_covariates)
        # When trained with target (1 channel) + past_covariates (13 channels) = 14 total channels
        # We need to create a wrapper that properly formats the input
        class TCNONNXWrapper(nn.Module):
            def __init__(self, tcn_module):
                super().__init__()
                self.tcn_module = tcn_module

            def forward(self, x):
                # Input: (batch, seq, features) where features=13 (all features including target)
                # DARTS TCN expects: (batch, channels, seq) where channels = target + covariates
                # Since covariates include the target already, we need 14 channels:
                # - First channel: target (Close price)
                # - Next 13 channels: all covariates (including Close)

                batch_size, seq_len, n_features = x.shape

                # Transpose to (batch, features, seq)
                x = x.transpose(1, 2)  # (batch, 13, seq)

                # Extract target (first channel is Close price)
                target = x[:, 0:1, :]  # (batch, 1, seq)

                # Concatenate target + all covariates to get 14 channels
                # This matches the training setup: target=Close, past_covariates=all 13 features
                x = torch.cat([target, x], dim=1)  # (batch, 14, seq)

                # Process through residual blocks
                for res_block in self.tcn_module.res_blocks:
                    x = res_block(x)

                # Transpose back to (batch, seq, features)
                x = x.transpose(1, 2)

                # Extract the last output_chunk_length predictions
                # Shape: (batch, output_chunk_length, target_size)
                x = x[:, -self.tcn_module.output_chunk_length:, :self.tcn_module.target_size]

                # If output_chunk_length=1, squeeze to (batch, target_size)
                if self.tcn_module.output_chunk_length == 1:
                    x = x.squeeze(1)

                return x

        wrapped_model = TCNONNXWrapper(torch_model)
        wrapped_model.eval()

        # Create dummy input in standard format (batch, sequence_length, n_features)
        sequence_length, n_features = input_shape
        dummy_input = torch.randn(1, sequence_length, n_features, dtype=torch.float32)

        print(f"Debug: Exporting TCN with input shape: {dummy_input.shape}")

        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )

        print(f"TCN model successfully converted to ONNX: {output_path}")
        return output_path

    except Exception as e:
        import traceback
        print(f"Warning: TCN ONNX conversion failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("The model can still be used with native DARTS inference.")
        return None


def convert_xgboost_to_onnx(
    pipeline,
    n_features: int,
    output_path: str,
    target_opset: int = 12
) -> str:
    """
    Convert XGBoost pipeline to ONNX format.

    Args:
        pipeline: Trained sklearn Pipeline with XGBoost
        n_features: Number of input features (flattened)
        output_path: Path to save ONNX model
        target_opset: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
    from xgboost import XGBRegressor

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Register XGBoost converter
        update_registered_converter(
            XGBRegressor,
            'XGBoostXGBRegressor',
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
            options={'nocl': [True, False]}
        )

        # Define initial types
        initial_type = [('input', FloatTensorType([None, n_features]))]
        target_opset_dict = {'': target_opset, 'ai.onnx.ml': 3}
        # Convert to ONNX
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=target_opset_dict
        )

        # Save ONNX model
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        print(f"XGBoost model successfully converted to ONNX: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error converting XGBoost to ONNX: {str(e)}")
        raise