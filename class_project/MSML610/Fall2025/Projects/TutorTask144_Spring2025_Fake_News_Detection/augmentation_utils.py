"""
augmentation_utils.py

Data augmentation module for expanding training data through text paraphrasing
and back-translation. Helps improve model robustness and performance.
"""

import logging
from typing import List, Dict, Tuple
from pathlib import Path
import json
import time

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class ParaphraseAugmenter:
    """Generate paraphrases using T5 model."""

    def __init__(self, model_name: str = "t5-small", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize paraphraser.

        Args:
            model_name: T5 model variant (t5-small, t5-base, t5-large)
            device: Device to run on
        """
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load T5 model and tokenizer."""
        logger.info(f"Loading {self.model_name} for paraphrasing...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def paraphrase(
        self,
        text: str,
        num_beams: int = 5,
        num_return_sequences: int = 3,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate paraphrases of input text.

        Args:
            text: Text to paraphrase
            num_beams: Beam search width
            num_return_sequences: Number of paraphrases to generate
            max_length: Maximum output length
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            List of paraphrases
        """
        input_text = f"paraphrase: {text} </s>"
        encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True
            )

        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return paraphrases

    def augment_batch(
        self,
        texts: List[str],
        num_paraphrases: int = 2,
        batch_size: int = 8
    ) -> Tuple[List[str], List[int]]:
        """
        Augment a batch of texts.

        Args:
            texts: List of texts to augment
            num_paraphrases: Number of paraphrases per text
            batch_size: Batch size for processing

        Returns:
            Tuple of (augmented_texts, original_indices)
        """
        augmented_texts = []
        original_indices = []

        for i, text in enumerate(texts):
            try:
                paraphrases = self.paraphrase(text, num_return_sequences=num_paraphrases)
                augmented_texts.extend(paraphrases)
                original_indices.extend([i] * num_paraphrases)
            except Exception as e:
                logger.warning(f"Failed to paraphrase text {i}: {str(e)}")
                augmented_texts.append(text)
                original_indices.append(i)

        return augmented_texts, original_indices


class BackTranslationAugmenter:
    """Generate back-translations using MarianMT models."""

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "fr",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize back-translator.

        Args:
            source_lang: Source language code (en)
            target_lang: Target language code (fr, de, es, etc.)
            device: Device to run on
        """
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.forward_tokenizer = None
        self.forward_model = None
        self.backward_tokenizer = None
        self.backward_model = None
        self._load_models()

    def _load_models(self):
        """Load MarianMT models for both directions."""
        logger.info(f"Loading back-translation models ({self.source_lang} <-> {self.target_lang})...")

        # Forward: source_lang -> target_lang
        forward_model_name = f"Helsinki-NLP/Opus-MT-{self.source_lang}-{self.target_lang}"
        self.forward_tokenizer = AutoTokenizer.from_pretrained(forward_model_name)
        self.forward_model = AutoModelForSeq2SeqLM.from_pretrained(forward_model_name).to(self.device)

        # Backward: target_lang -> source_lang
        backward_model_name = f"Helsinki-NLP/Opus-MT-{self.target_lang}-{self.source_lang}"
        self.backward_tokenizer = AutoTokenizer.from_pretrained(backward_model_name)
        self.backward_model = AutoModelForSeq2SeqLM.from_pretrained(backward_model_name).to(self.device)

        logger.info(f"Back-translation models loaded on {self.device}")

    def back_translate(self, text: str, num_beams: int = 5) -> str:
        """
        Generate back-translation: EN -> FR -> EN

        Args:
            text: Text to back-translate
            num_beams: Beam search width

        Returns:
            Back-translated text
        """
        # Forward: EN -> FR
        forward_encoding = self.forward_tokenizer.encode_plus(
            text,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            forward_ids = self.forward_model.generate(
                input_ids=forward_encoding["input_ids"].to(self.device),
                attention_mask=forward_encoding["attention_mask"].to(self.device),
                num_beams=num_beams,
                max_length=256
            )

        intermediate = self.forward_tokenizer.decode(forward_ids[0], skip_special_tokens=True)

        # Backward: FR -> EN
        backward_encoding = self.backward_tokenizer.encode_plus(
            intermediate,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            backward_ids = self.backward_model.generate(
                input_ids=backward_encoding["input_ids"].to(self.device),
                attention_mask=backward_encoding["attention_mask"].to(self.device),
                num_beams=num_beams,
                max_length=256
            )

        back_translated = self.backward_tokenizer.decode(backward_ids[0], skip_special_tokens=True)

        return back_translated

    def augment_batch(
        self,
        texts: List[str],
        num_back_translations: int = 1
    ) -> Tuple[List[str], List[int]]:
        """
        Back-translate a batch of texts.

        Args:
            texts: List of texts to augment
            num_back_translations: Number of back-translations per text

        Returns:
            Tuple of (augmented_texts, original_indices)
        """
        augmented_texts = []
        original_indices = []

        for i, text in enumerate(texts):
            try:
                for j in range(num_back_translations):
                    back_trans = self.back_translate(text)
                    augmented_texts.append(back_trans)
                    original_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to back-translate text {i}: {str(e)}")
                augmented_texts.append(text)
                original_indices.append(i)

        return augmented_texts, original_indices


class DataAugmentationPipeline:
    """Combined data augmentation pipeline using multiple techniques."""

    def __init__(
        self,
        use_paraphrase: bool = True,
        use_back_translation: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize augmentation pipeline.

        Args:
            use_paraphrase: Enable paraphrase augmentation
            use_back_translation: Enable back-translation augmentation
            device: Device to run on
        """
        self.device = device
        self.paraphraser = None
        self.back_translator = None

        if use_paraphrase:
            self.paraphraser = ParaphraseAugmenter(model_name="t5-small", device=device)

        if use_back_translation:
            self.back_translator = BackTranslationAugmenter(
                source_lang="en",
                target_lang="fr",
                device=device
            )

    def augment(
        self,
        texts: List[str],
        labels: List[int],
        augmentation_factor: float = 1.0,
        methods: List[str] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Augment training data using multiple techniques.

        Args:
            texts: List of texts
            labels: List of labels
            augmentation_factor: How much to augment (0.5 = 50% more data, 1.0 = 2x data)
            methods: List of augmentation methods (['paraphrase', 'back_translate'])

        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if methods is None:
            methods = ['paraphrase', 'back_translate']

        augmented_texts = texts.copy()
        augmented_labels = labels.copy()

        target_size = int(len(texts) * augmentation_factor)
        current_size = len(texts)
        texts_to_add = target_size - current_size

        if texts_to_add <= 0:
            logger.info("No augmentation needed")
            return augmented_texts, augmented_labels

        texts_to_augment = texts[:texts_to_add]
        labels_to_augment = labels[:texts_to_add]

        logger.info(f"Augmenting {texts_to_add} texts to reach target size of {target_size}")

        # Rotate through augmentation methods
        for idx, (text, label) in enumerate(zip(texts_to_augment, labels_to_augment)):
            method = methods[idx % len(methods)]

            try:
                if method == 'paraphrase' and self.paraphraser is not None:
                    paraphrases = self.paraphraser.paraphrase(text, num_return_sequences=1)
                    if paraphrases:
                        augmented_texts.append(paraphrases[0])
                        augmented_labels.append(label)

                elif method == 'back_translate' and self.back_translator is not None:
                    back_trans = self.back_translator.back_translate(text)
                    augmented_texts.append(back_trans)
                    augmented_labels.append(label)

            except Exception as e:
                logger.warning(f"Augmentation failed for text {idx}: {str(e)}")
                continue

        logger.info(
            f"Augmentation complete: {len(texts)} -> {len(augmented_texts)} texts "
            f"(+{len(augmented_texts) - len(texts)} samples)"
        )

        return augmented_texts, augmented_labels

    def save_augmented_data(
        self,
        texts: List[str],
        labels: List[int],
        output_dir: Path
    ):
        """Save augmented data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'texts': texts,
            'labels': labels,
            'metadata': {
                'total_samples': len(texts),
                'num_fake': sum(1 for l in labels if l == 0),
                'num_real': sum(1 for l in labels if l == 1)
            }
        }

        with open(output_dir / 'augmented_data.json', 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Augmented data saved to {output_dir}/augmented_data.json")

    @staticmethod
    def load_augmented_data(filepath: Path) -> Tuple[List[str], List[int]]:
        """Load augmented data from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return data['texts'], data['labels']
