"""
Main utilities for the ONNX Fake News Detection project.

This module provides:
1. Dataset loading and preprocessing utilities
2. LSTM-based text classification (TensorFlow / Keras)
3. Conversion of trained models to ONNX
4. Inference using ONNX Runtime
5. (Bonus) DistilBERT fine-tuning and ONNX conversion
6. (Bonus) FastAPI app factory for serving ONNX models

"""

# ===============================================================
# Standard library imports
# ===============================================================
import os
import pickle
from typing import List, Dict, Tuple, Optional

# ===============================================================
# Third-party imports
# ===============================================================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import tensorflow as tf

# ---- Safety: disabled XLA / GPU to avoid JIT issues on my PC
tf.config.optimizer.set_jit(False)
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

import tf2onnx
import onnx
import onnxruntime as ort

# ===============================================================
# Bonus: HuggingFace + FastAPI
# ===============================================================
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ===============================================================
# Global paths and hyperparameters
# ===============================================================

DATA_DIR = "data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
REAL_PATH = os.path.join(DATA_DIR, "True.csv")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ----- LSTM configuration
LSTM_TOKENIZER_PATH = os.path.join(MODELS_DIR, "lstm_tokenizer.pkl")
LSTM_KERAS_PATH = os.path.join(MODELS_DIR, "lstm_fake_news.keras")
LSTM_ONNX_PATH = os.path.join(MODELS_DIR, "lstm_fake_news.onnx")

LSTM_MAX_WORDS = 20000
LSTM_MAX_LEN = 200
LSTM_EMBED_DIM = 128
LSTM_UNITS = 128
LSTM_BATCH_SIZE = 128
LSTM_EPOCHS = 1  # increase later if desired

# ----- DistilBERT configuration (bonus)

DISTILBERT_ONNX_PATH = os.path.join(MODELS_DIR, "distilbert_fake_news.onnx")


# ===============================================================
# Data loading and metrics
# ===============================================================

def load_fake_real_news(
    fake_path: str = FAKE_PATH,
    real_path: str = REAL_PATH,
) -> pd.DataFrame:
    """
    Load Kaggle Fake/True news CSV files and return a shuffled DataFrame.

    Output columns:
        - text  : combined title + article text
        - label : 0 = fake, 1 = real
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Missing file: {fake_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Missing file: {real_path}")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    def combine(row) -> str:
        parts = []
        for col in ("title", "text"):
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return " ".join(parts).strip()

    fake_texts = fake_df.apply(combine, axis=1)
    real_texts = real_df.apply(combine, axis=1)

    df = pd.concat(
        [
            pd.DataFrame({"text": fake_texts, "label": 0}),
            pd.DataFrame({"text": real_texts, "label": 1}),
        ],
        ignore_index=True,
    )

    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


# ===============================================================
# LSTM utilities
# ===============================================================

def tokenize_and_pad(
    texts: List[str],
    tokenizer: Optional[Tokenizer] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, Tokenizer]:
    """
    Tokenize raw text and pad to fixed length.
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=LSTM_MAX_WORDS, oov_token="<OOV>")
        fit = True

    if fit:
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=LSTM_MAX_LEN,
        padding="post",
        truncating="post",
    )
    return padded, tokenizer


def build_lstm_model() -> tf.keras.Model:
    """
    Build a BiLSTM-based binary classifier.
    """
    model = models.Sequential(
        [
            layers.Embedding(
                LSTM_MAX_WORDS,
                LSTM_EMBED_DIM,
                input_length=LSTM_MAX_LEN,
            ),
            layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
# Building the LSTM model architecture
# - Embedding layer: converts token IDs to dense vectors
# - BiDirectional LSTM: processes text in both directions for better context understanding
# - Dense(64) + Dropout: fully connected layer with regularization to prevent overfitting
# - Dense(1) + Sigmoid: output layer for binary classification (0=fake, 1=real)


def train_lstm_model(
    num_samples: Optional[int] = None,
) -> Dict[str, object]:
    """
    Train the LSTM model and export it as TensorFlow SavedModel.
    """
    tf.config.run_functions_eagerly(True)

    df = load_fake_real_news()
    if num_samples:
        df = df.head(num_samples)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        # Load fake and real news data
        df["text"].tolist(),
        # Split data: 85% training, 15% testing with stratification for balanced distribution
        df["label"].values,
        test_size=0.15,
        stratify=df["label"].values,
        random_state=42,
    )

    X_train, tokenizer = tokenize_and_pad(X_train_texts, fit=True)
    # Tokenize training texts and fit the tokenizer on the training vocabulary
    X_test, _ = tokenize_and_pad(X_test_texts, tokenizer=tokenizer)
# Tokenize test texts using the same fitted tokenizer to ensure consistency

    with open(LSTM_TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
# Save the tokenizer for future inference

    model = build_lstm_model()
    history = model.fit(
        # Train the model on tokenized training data with validation on test data
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=1,
    )

    model.save(LSTM_KERAS_PATH)
    print(f">>> Saved Keras model to {LSTM_KERAS_PATH}")

    # Export trained model to TensorFlow keras format for ONNX conversion

    preds = (model.predict(X_test).reshape(-1) > 0.5).astype(int)
    # Generate predictions: threshold at 0.5 to classify as fake (0) or real (1)
    metrics = compute_classification_metrics(y_test, preds)

    return {
        "history": history.history,
        "metrics": metrics,
    }


def convert_lstm_to_onnx() -> str:
    """
    Convert trained LSTM Keras model to ONNX.

    This implementation is compatible with:
    - Keras 3
    - tf2onnx
    - Sequential models
    """

    print(">>> Loading Keras model...")
    keras_model = tf.keras.models.load_model(LSTM_KERAS_PATH, compile=False)
    input_ids = tf.keras.Input(
        shape=(LSTM_MAX_LEN,),
        dtype=tf.int32,
        name="input_ids"
    )

    outputs = keras_model(input_ids)
    functional_model = tf.keras.Model(
        inputs=input_ids,
        outputs=outputs,
        name="lstm_functional"
    )

    print(">>> Converting to ONNX...")

    model_proto, _ = tf2onnx.convert.from_keras(
        functional_model,
        input_signature=(
            tf.TensorSpec(
                (None, LSTM_MAX_LEN),
                tf.int32,
                name="input_ids"
            ),
        ),
        opset=13,
    )

    onnx.save(model_proto, LSTM_ONNX_PATH)
    print(f">>> Saved ONNX model to {LSTM_ONNX_PATH}")

    return LSTM_ONNX_PATH




def predict_lstm_onnx(texts: List[str]) -> List[Dict[str, float]]:
    """
    Run inference using ONNX Runtime on LSTM model.
    """
    with open(LSTM_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f) # Load the saved tokenizer used during training

    seqs, _ = tokenize_and_pad(texts, tokenizer=tokenizer)
    # Tokenize input texts using the training tokenizer
    seqs = seqs.astype("int32")
# Ensure sequence data is int32 type as expected by ONNX model

    sess = ort.InferenceSession(LSTM_ONNX_PATH)
    # Initialize ONNX Runtime session with the trained model
    input_name = sess.get_inputs()[0].name
    # Get the name of the input tensor from the ONNX model
    probs = sess.run(None, {input_name: seqs.astype("int32")})[0].reshape(-1)
# Run inference and extract probability scores

    return [
        # Return prediction results including text, binary label (0/1), and confidence score
        {"text": t, "label": int(p > 0.5), "score": float(p)}
        for t, p in zip(texts, probs)
    ]



# ===============================================================
# FastAPI serving (bonus)
# ===============================================================

class FakeNewsRequest(BaseModel):
    text: str


def create_fastapi_app(model_type: str = "lstm") -> FastAPI:
    """
    Create a FastAPI app serving either LSTM or DistilBERT ONNX model.
    """
    app = FastAPI(title="ONNX Fake News Detection API")
# Initialize FastAPI application with descriptive title

    if model_type == "lstm":
        # Load LSTM-specific resources for inference
        # Initialize ONNX Runtime session with trained LSTM model
        # Load the tokenizer used during LSTM training
        sess = ort.InferenceSession(LSTM_ONNX_PATH)
        with open(LSTM_TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

    @app.post("/predict")
    # Define HTTP POST endpoint for making predictions
    async def predict(req: FakeNewsRequest):
        # Handle incoming prediction request with news text
        if not req.text.strip():
            raise HTTPException(400, "Empty input")
# Validate that input text is not empty
# Execute LSTM inference

        if model_type == "lstm":
            seqs, _ = tokenize_and_pad([req.text], tokenizer)
            # Tokenize input text using the same tokenizer as training
            probs = sess.run(None, {sess.get_inputs()[0].name: seqs.astype("int32")})[0]
            # Run ONNX inference and get probability score
            prob = float(probs[0][0])
            # Extract scalar probability value
            return {"label": int(prob > 0.5), "score": prob}
# Return classification result: label (0=fake, 1=real) and confidence score

        raise HTTPException(500, "Model not available")

    return app
