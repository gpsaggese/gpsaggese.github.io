import numpy as np
import os
import json
from typing import Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import onnxruntime as ort


def convert_to_onnx(model_path: str, onnx_path: str, opset: int = 13) -> str:
    """
    Convert Keras model to ONNX format.

    Args:
        model_path: Path to Keras model (.h5 or SavedModel)
        onnx_path: Path to save ONNX model
        opset: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model = keras.models.load_model(model_path)
    input_signature = [tf.TensorSpec(model.input.shape, tf.float32, name='input')]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=onnx_path
    )

    return onnx_path


def verify_onnx(onnx_path: str) -> Dict[str, Any]:
    """
    Verify ONNX model using onnx.checker.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Verification results
    """
    model = onnx_utils.load(onnx_path)

    try:
        onnx_utils.checker.check_model(model)
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