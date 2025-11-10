"""
Handles output formatting and evaluation metrics.
"""

from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(preds, labels):
    """Compute accuracy and confusion matrix."""
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return acc, cm
