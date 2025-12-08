

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class IntentClassifier:
    """
    Zero-shot multilingual intent classifier using XLM-RoBERTa Large.

    MUCH stronger than BART-large for semantic understanding,
    supports 100+ languages, fits ~2GB RAM, ideal for general assistants.
    """

    def __init__(self):
        print("[IntentClassifier] Loading XLM-RoBERTa-Large for zero-shot intent classification...")

        self.model_name = "joeddav/xlm-roberta-large-xnli"   # Multilingual NLI head
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Expandable intent list for a general-purpose assistant
        self.labels = [
            "greeting",
            "goodbye",
            "help_request",
            "order_issue",
            "refund_request",
            "device_control",
            "search_query",
            "chit_chat",
            "general_question",
            "product_question",
            "math_question",
            "translation_request",
            "travel_information",
            "medical_information",
            "technical_support",
            "programming_help",
            "task_management"
        ]

    def predict_intent(self, text: str) -> str:
        """
        Zero-shot intent classification using XNLI-style NLI ranking.
        Returns the best matching intent.
        """

        if not text:
            return "help_request"

        best_label = None
        best_score = -1

        for label in self.labels:
            hypothesis = f"This text is about {label}."

            inputs = self.tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True
            )

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1)

                # XNLI labels: 0 = contradiction, 1 = neutral, 2 = entailment
                entail_score = probs[0][2].item()

            if entail_score > best_score:
                best_score = entail_score
                best_label = label

        return best_label
