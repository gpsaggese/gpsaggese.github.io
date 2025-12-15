"""
MCP Server Class for BERT Fake News Detection

This module implements the BERTMCPServer class that provides a complete
Model Context Protocol (MCP) server for the BERT fake news detection model.
"""

import time
import numpy as np
from typing import Dict, List, Any

import torch
from transformers import BertTokenizer, BertForSequenceClassification

import bert_utils


class BERTMCPServer:
    """
    MCP Server for BERT Fake News Detection.

    This server wraps a fine-tuned BERT model and exposes a clean API for
    running fake-news predictions. It’s built to handle both single requests
    and batch jobs, while also keeping track of past predictions and basic
    model stats for monitoring and debugging.

    Attributes:
        model: Fine-tuned BERT model for classification
        tokenizer: BERT tokenizer for text encoding
        model_id: Unique identifier for the model
        prediction_history: List of all predictions made by the server
        metadata: Dictionary containing model information and metrics
    """

    def __init__(self, model: BertForSequenceClassification, tokenizer: BertTokenizer, model_id: str = "bert_fake_news"):
       
        # Initialize MCP server with loaded model and tokenizer.
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.prediction_history = []

        # Model metadata
        self.metadata = {
            "id": model_id,
            "type": "bert-fake-news-detector",
            "version": "1.0",
            "training_accuracy": 0.9991,
            "unseen_accuracy": 0.8474,
            "precision": 0.8213,
            "recall": 0.8797,
            "f1_score": 0.8495,
            "roc_auc": 0.9360,
            "parameters": 110_000_000,
            "model_type": "bert-base-uncased",
            "test_samples": 6_734,
            "unseen_samples": 64_951,
            "gpu_acceleration": True
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [{
            "id": self.model_id,
            "type": self.metadata["type"],
            "accuracy": self.metadata["unseen_accuracy"],
            "version": self.metadata["version"]
        }]

    def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        if model_id is None or model_id == self.model_id:
            return self.metadata
        else:
            raise ValueError(f"Model {model_id} not found")

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: text must be non-empty string"}

        start_time = time.time()

        try:
            label, confidence = bert_utils.predict_text(text, self.model, self.tokenizer)

            processing_time = time.time() - start_time
            response = {
                "model_id": self.model_id,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "prediction": {
                    "label": int(label),
                    "class": "REAL" if label == 1 else "FAKE",
                    "confidence": float(confidence),
                    "confidence_percent": f"{confidence*100:.2f}%"
                },
                "metadata": {
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "text_length": len(text),
                    "timestamp": time.time()
                }
            }

            self.prediction_history.append(response)

            return response
        except Exception as e:
            return {"error": str(e)}

    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        if not texts or not isinstance(texts, list):
            return {"error": "Invalid input: texts must be non-empty list"}

        start_time = time.time()
        predictions = []

        for text in texts:
            pred = self.predict(text)
            if "error" not in pred:
                predictions.append(pred)

        processing_time = time.time() - start_time

        # Calculate statistics
        real_count = sum(1 for p in predictions if p["prediction"]["label"] == 1)
        fake_count = sum(1 for p in predictions if p["prediction"]["label"] == 0)
        avg_confidence = np.mean([p["prediction"]["confidence"] for p in predictions]) if predictions else 0.0

        return {
            "model_id": self.model_id,
            "total": len(predictions),
            "real_count": real_count,
            "fake_count": fake_count,
            "real_percent": f"{100*real_count/len(predictions):.1f}%" if predictions else "0%",
            "fake_percent": f"{100*fake_count/len(predictions):.1f}%" if predictions else "0%",
            "avg_confidence": float(avg_confidence),
            "predictions": predictions,
            "metadata": {
                "total_processing_time_s": round(processing_time, 2),
                "avg_time_per_article_ms": round(processing_time/len(predictions)*1000, 2) if predictions else 0,
                "timestamp": time.time()
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        if not self.prediction_history:
            return {
                "total_predictions": 0,
                "message": "No predictions made yet"
            }

        predictions = [p["prediction"] for p in self.prediction_history if "error" not in p]

        if not predictions:
            return {
                "total_predictions": 0,
                "message": "No successful predictions"
            }

        return {
            "total_predictions": len(predictions),
            "real_predictions": sum(1 for p in predictions if p["label"] == 1),
            "fake_predictions": sum(1 for p in predictions if p["label"] == 0),
            "avg_confidence": float(np.mean([p["confidence"] for p in predictions])),
            "min_confidence": float(np.min([p["confidence"] for p in predictions])),
            "max_confidence": float(np.max([p["confidence"] for p in predictions]))
        }

    def clear_history(self) -> None:
        self.prediction_history = []


# Example usage
if __name__ == "__main__":
    print("MCP Server Class Module")
    print("\nThis module provides the BERTMCPServer class for BERT fake news detection.")
    print("\nUsage:")
    print("  from mcp_server_class import BERTMCPServer")
    print("  import bert_utils")
    print("")
    print("  # Load model")
    print("  model, tokenizer = bert_utils.load_model('models/bert_fake_news')")
    print("")
    print("  # Initialize server")
    print("  server = BERTMCPServer(model, tokenizer)")
    print("")
    print("  # Make predictions")
    print("  result = server.predict('Article text here...')")
    print("  results = server.predict_batch(['Article 1...', 'Article 2...'])")
    print("")
    print("  # Get statistics")
    print("  stats = server.get_statistics()")
    print("\nSee MCP_Server_Implementation.ipynb for complete examples.")
