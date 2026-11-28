
"""Tests for the financial sentiment analysis utilities."""

import importlib.util
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

UTILS_PATH = PROJECT_DIR / (
    "UmdTask791_MSML610_Fall2026_"
    "NLTK_Sentiment_Analysis_of_Financial_News_utils.py"
)

spec = importlib.util.spec_from_file_location(
    "financial_sentiment_utils",
    UTILS_PATH,
)
financial_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(financial_utils)


class TestFinancialSentimentUtils(unittest.TestCase):

    def test_normalize_headline(self):
        result = financial_utils.normalize_headline(
            "  Company   Reports   Higher   Revenue  "
        )
        self.assertEqual(result, "company reports higher revenue")

    def test_clean_headline(self):
        result = financial_utils.clean_headline(
            "  Profit   increased   by  10%  "
        )
        self.assertEqual(result, "Profit increased by 10%")

    def test_invalid_headline(self):
        with self.assertRaises(TypeError):
            financial_utils.normalize_headline(None)

    def test_perfect_predictions(self):
        actual = ["positive", "negative", "neutral"]
        metrics = financial_utils.evaluate_predictions(
            actual, actual
        )
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            financial_utils.evaluate_predictions(
                ["positive", "negative"],
                ["positive"],
            )

    def test_empty_predictions(self):
        with self.assertRaises(ValueError):
            financial_utils.evaluate_predictions([], [])

    def test_model_comparison(self):
        results = {
            "Model A": {"accuracy": 0.7, "macro_f1": 0.6},
            "Model B": {"accuracy": 0.8, "macro_f1": 0.75},
        }
        comparison = financial_utils.build_model_comparison(
            results
        )
        self.assertEqual(comparison.iloc[0]["Model"], "Model B")
        self.assertEqual(len(comparison), 2)


if __name__ == "__main__":
    unittest.main()
