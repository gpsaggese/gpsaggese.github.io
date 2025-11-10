# 🧠 ONNX API Overview

This document describes how the ONNX Fake News Detection API interacts with the model.

## Components
- **ONNX Model Loader** → Loads serialized model
- **Inference Session** → Executes ONNX graph
- **Preprocessing Module** → Prepares text for model
- **Post-processing Module** → Converts logits to class labels
