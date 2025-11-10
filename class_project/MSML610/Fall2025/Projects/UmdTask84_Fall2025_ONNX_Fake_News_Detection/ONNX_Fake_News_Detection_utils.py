"""
Utility functions for ONNX Fake News Detection Project
Author: Keshav Naram
"""

import onnxruntime as ort
import numpy as np
import pandas as pd

def load_onnx_model(model_path: str):
    """Load an ONNX model from the given path."""
    try:
        session = ort.InferenceSession(model_path)
        print(f" Model loaded from {model_path}")
        return session
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

def predict(session, input_data: np.ndarray):
    """Run inference using the ONNX model."""
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    return outputs[0]
