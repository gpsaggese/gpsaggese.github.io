# ONNX Fake News Detection: Example Application Logic

## 1. Intent of the Example
The primary goal of this application is to demonstrate a **production-ready machine learning lifecycle**. While training is often the focus of data science, this example prioritizes **model portability** and **inference performance**. 

By comparing traditional deep learning frameworks (TensorFlow/PyTorch) against the [ONNX Runtime](https://onnxruntime.ai/), I demonstrate how to bridge the gap between a research environment and a high-efficiency deployment environment.

## 2. Design Decisions & Reasoning

### Hybrid Model Architecture
The application evaluates two distinct architectural approaches:
* **Bi-LSTM (TensorFlow):** Chosen for its balance between computational efficiency and the ability to capture long-term dependencies in text. This serves as our primary API model.
* **DistilBERT (HuggingFace):** Included to demonstrate how even state-of-the-art [Transformer models](https://huggingface.co/docs/transformers/model_doc/distilbert) can be optimized for CPU deployment using ONNX.

### Why ONNX for Deployment?
The transition to ONNX was driven by three key factors:
1.  **Framework Decoupling:** In production, we want to avoid installing the massive `tensorflow` or `torch` libraries. [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) is a lightweight C++ engine with a Python wrapper that runs independently of the training framework.
2.  **Inference Speedup:** Native frameworks carry overhead designed for training (like gradient tracking). ONNX graphs are "frozen" and optimized (via graph fusion and constant folding), resulting in significantly lower latency.
3.  **Accuracy Preservation:** A critical design requirement was that the conversion must be "lossless." The example application includes a parity check to ensure that $Accuracy_{Native} \approx Accuracy_{ONNX}$.

### Memory Stability & Batching
During the evaluation of the DistilBERT model, I encountered "Kernel Dying" issues due to memory exhaustion. 
* **Decision:** We implemented a **Batched Inference Loop**. 
* **Reasoning:** Processing 500+ news articles simultaneously creates a massive memory spike. By batching (e.g., 16 samples at a time) and using [explicit garbage collection](https://docs.python.org/3/library/gc.html), we maintain a constant, low-memory footprint, ensuring the application remains stable on standard CPU hardware.

## 3. Workflow Summary
The application follows a structured four-stage pipeline:
1.  **Data Ingestion:** Merging and shuffling the Kaggle Fake/Real news CSVs into a unified stream.
2.  **Native Training:** Fine-tuning weights in a high-level framework.
3.  **Graph Export:** Converting the trained model into a `.onnx` binary using [tf2onnx](https://github.com/onnx/tensorflow-onnx) or the [Torch ONNX exporter](https://pytorch.org/docs/stable/onnx.html).
4.  **Performance Benchmarking:** Executing a side-by-side comparison of latency and accuracy to validate the deployment readiness.
5. **Fast API demo**: Demonstrating how to use Fast API to expose ONNX model for prediction.
