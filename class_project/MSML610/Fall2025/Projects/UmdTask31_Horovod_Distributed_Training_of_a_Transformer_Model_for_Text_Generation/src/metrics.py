"""
Metrics computation for language model evaluation.

Includes perplexity, BLEU, and ROUGE scores.
"""

import math
from typing import List, Optional, Dict, Tuple

import torch
import numpy as np


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss value.
        
    Returns:
        Perplexity value.
    """
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float('inf')
    
    return perplexity


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Target labels of shape (batch, seq_len).
        pad_token_id: Padding token ID to ignore.
        
    Returns:
        Accuracy as a float between 0 and 1.
    """
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens (ignore positions marked with ignore_index)
    mask = (labels != ignore_index)
    
    # Compute accuracy only on non-padding tokens
    correct = (predictions == labels) & mask
    mask_sum = mask.sum().item()
    
    # Avoid division by zero
    if mask_sum == 0:
        return 0.0
    
    accuracy = correct.sum().item() / mask_sum
    
    return accuracy


def compute_accuracy_for_allreduce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Tuple[float, float]:
    """
    Compute accuracy numerator and denominator for distributed averaging.
    
    Returns (correct_count, total_count) so they can be allreduced separately
    and then divided to get the true average across all ranks.
    
    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Target labels of shape (batch, seq_len).
        pad_token_id: Padding token ID to ignore.
        
    Returns:
        Tuple of (correct_count, total_count) as floats.
    """
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens (ignore positions marked with ignore_index)
    mask = (labels != ignore_index)
    
    # Compute accuracy only on non-padding tokens
    correct = (predictions == labels) & mask
    correct_count = correct.sum().item()
    total_count = mask.sum().item()
    
    return float(correct_count), float(total_count)


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score for generated text.
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        max_n: Maximum n-gram order (default: 4 for BLEU-4).
        
    Returns:
        Dictionary with BLEU scores.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        smoothing = SmoothingFunction().method1
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]
            
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=smoothing
            )
            bleu_scores.append(score)
        
        avg_bleu = np.mean(bleu_scores)
        
        return {
            'bleu': avg_bleu,
            'bleu_samples': len(bleu_scores)
        }
        
    except ImportError:
        print("[WARN] NLTK not available. Skipping BLEU computation.")
        return {'bleu': 0.0, 'bleu_samples': 0}
    except Exception as e:
        print(f"[WARN] Failed to compute BLEU: {e}")
        return {'bleu': 0.0, 'bleu_samples': 0}


def compute_rouge_score(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores for generated text.
    
    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        
    Returns:
        Dictionary with ROUGE scores.
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge_samples': len(rouge1_scores)
        }
        
    except ImportError:
        print("[WARN] rouge-score package not available. Skipping ROUGE computation.")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rouge_samples': 0}
    except Exception as e:
        print(f"[WARN] Failed to compute ROUGE: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rouge_samples': 0}


class MetricsTracker:
    """
    Track metrics during training and evaluation.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.perplexities = []
        self.accuracies = []
    
    def update(self, loss: float, logits: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, accuracy: Optional[float] = None):
        """
        Update metrics with new batch.
        
        Args:
            loss: Loss value.
            logits: Model output logits (optional).
            labels: Target labels (optional).
            accuracy: Pre-computed accuracy (optional, for distributed training).
        """
        self.losses.append(loss)
        self.perplexities.append(compute_perplexity(loss))
        
        if accuracy is not None:
            # Use provided accuracy (already averaged across ranks)
            self.accuracies.append(accuracy)
        elif logits is not None and labels is not None:
            # Compute locally (for backward compatibility)
            accuracy = compute_accuracy(logits, labels)
            self.accuracies.append(accuracy)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average of all tracked metrics.
        
        Returns:
            Dictionary of average metrics.
        """
        metrics = {
            'loss': np.mean(self.losses) if self.losses else 0.0,
            'perplexity': np.mean(self.perplexities) if self.perplexities else 0.0,
        }
        
        if self.accuracies:
            metrics['accuracy'] = np.mean(self.accuracies)
        
        return metrics
