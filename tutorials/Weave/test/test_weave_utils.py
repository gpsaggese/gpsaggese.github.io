"""
Import as:

import tutorials.Weave.test.test_weave_utils as twetewuti
"""

import logging

import pytest

pytest.importorskip("weave")

import helpers.hunit_test as hunitest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tutorials.Weave.weave_utils as tweweuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_keyword_tone
# #############################################################################


class Test_keyword_tone(hunitest.TestCase):
    """
    Test keyword-based monetary policy classification.
    """

    def test1(self) -> None:
        """
        Test a hawkish sentence.
        """
        # Prepare inputs.
        sentence = "The committee will raise interest rates."
        # Run test.
        actual = tweweuti.keyword_tone(
            sentence, ["raise"], ["lower"]
        )
        # Check outputs.
        self.assertEqual(actual["label"], "hawkish")
        self.assertEqual(actual["hawkish_hits"], 1)
        self.assertEqual(actual["dovish_hits"], 0)

    def test2(self) -> None:
        """
        Test a dovish sentence.
        """
        # Run test.
        actual = tweweuti.keyword_tone(
            "The committee will lower rates.",
            ["raise"],
            ["lower"],
        )
        # Check outputs.
        self.assertEqual(actual["label"], "dovish")

    def test3(self) -> None:
        """
        Test a sentence without matching keywords.
        """
        # Run test.
        actual = tweweuti.keyword_tone(
            "Economic conditions remain unchanged.",
            ["raise"],
            ["lower"],
        )
        # Check outputs.
        self.assertEqual(actual["label"], "neutral")

    def test4(self) -> None:
        """
        Test a sentence with equal hawkish and dovish hits.
        """
        # Run test.
        actual = tweweuti.keyword_tone(
            "Rates may rise or fall.",
            ["rise"],
            ["fall"],
        )
        # Check outputs.
        self.assertEqual(actual["label"], "neutral")


# #############################################################################
# Test_training_data_digest
# #############################################################################


class Test_training_data_digest(hunitest.TestCase):
    """
    Test training data fingerprints.
    """

    def test1(self) -> None:
        """
        Test that identical data produces identical fingerprints.
        """
        # Prepare inputs.
        data = pd.DataFrame(
            {"sentence": ["Lower rates."], "label": [0]}
        )
        # Run test.
        digest1 = tweweuti.training_data_digest(data)
        digest2 = tweweuti.training_data_digest(data.copy())
        # Check outputs.
        self.assertEqual(digest1, digest2)
        self.assertEqual(len(digest1), 64)

    def test2(self) -> None:
        """
        Test that changing a label changes the fingerprint.
        """
        # Prepare inputs.
        data = pd.DataFrame(
            {"sentence": ["Lower rates."], "label": [0]}
        )
        modified = data.copy()
        modified.loc[0, "label"] = 1
        # Run test.
        digest1 = tweweuti.training_data_digest(data)
        digest2 = tweweuti.training_data_digest(modified)
        # Check outputs.
        self.assertNotEqual(digest1, digest2)


# #############################################################################
# TestTfidfLogRegModel
# #############################################################################


class TestTfidfLogRegModel(hunitest.TestCase):
    """
    Test TF-IDF logistic regression classification.
    """

    def _make_training_data(self) -> pd.DataFrame:
        """
        Create a small synthetic classification dataset.

        :return: labeled training sentences
        """
        return pd.DataFrame(
            {
                "sentence": [
                    "Lower interest rates.",
                    "Lower rates and ease policy.",
                    "Raise interest rates.",
                    "Raise rates and tighten policy.",
                    "Conditions remain unchanged.",
                    "Economic conditions remain stable.",
                ],
                "label": [0, 0, 1, 1, 2, 2],
            }
        )

    def _make_model(
        self, data: pd.DataFrame
    ) -> tweweuti.TfidfLogRegModel:
        """
        Create a model with the matching training fingerprint.

        :param data: labeled training sentences
        :return: unfitted classification model
        """
        return tweweuti.TfidfLogRegModel(
            name="test-tfidf",
            training_digest=tweweuti.training_data_digest(data),
        )

    def test1(self) -> None:
        """
        Test fitting and predicting a valid label.
        """
        # Prepare inputs.
        data = self._make_training_data()
        model = self._make_model(data)
        # Run test.
        model.fit(data)
        actual = model.predict("Raise interest rates.")
        # Check outputs.
        self.assertIn(actual["label"], [0, 1, 2])
        self.assertEqual(
            actual["label_name"],
            tweweuti.LABEL_NAMES[actual["label"]],
        )
        self.assertGreaterEqual(actual["confidence"], 0.0)
        self.assertLessEqual(actual["confidence"], 1.0)

    def test2(self) -> None:
        """
        Test that prediction requires a fitted model.
        """
        # Prepare inputs.
        data = self._make_training_data()
        model = self._make_model(data)
        # Run test and check outputs.
        with self.assertRaises(RuntimeError):
            model.predict("Raise interest rates.")

    def test3(self) -> None:
        """
        Test consistency between individual and batch predictions.
        """
        # Prepare inputs.
        data = self._make_training_data()
        model = self._make_model(data)
        model.fit(data)
        sentences = data["sentence"].tolist()
        # Run test.
        actual = model.predict_batch(sentences)
        expected = [
            model.predict(sentence)["label"]
            for sentence in sentences
        ]
        # Check outputs.
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(sentences))

    def test4(self) -> None:
        """
        Test that a balanced model can be fitted.
        """
        # Prepare inputs.
        data = self._make_training_data()
        model = tweweuti.TfidfLogRegModel(
            name="test-balanced",
            training_digest=tweweuti.training_data_digest(data),
            class_weight="balanced",
        )
        # Run test.
        model.fit(data)
        actual = model.predict_batch(data["sentence"].tolist())
        # Check outputs.
        self.assertEqual(len(actual), len(data))
        self.assertTrue(set(actual).issubset({0, 1, 2}))


# #############################################################################
# Test_accuracy_scorer
# #############################################################################


class Test_accuracy_scorer(hunitest.TestCase):
    """
    Test the classification accuracy scorer.
    """

    def test1(self) -> None:
        """
        Test a correct prediction.
        """
        # Run test.
        actual = tweweuti.accuracy_scorer(1, {"label": 1})
        # Check outputs.
        self.assertTrue(actual["correct"])

    def test2(self) -> None:
        """
        Test an incorrect prediction.
        """
        # Run test.
        actual = tweweuti.accuracy_scorer(1, {"label": 0})
        # Check outputs.
        self.assertFalse(actual["correct"])


# #############################################################################
# TestMacroF1Scorer
# #############################################################################


class TestMacroF1Scorer(hunitest.TestCase):
    """
    Test macro-F1 aggregation.
    """

    def test1(self) -> None:
        """
        Test perfect classification.
        """
        # Prepare inputs.
        scorer = tweweuti.MacroF1Scorer()
        rows = [
            {"true_label": label, "predicted_label": label}
            for label in [0, 1, 2]
        ]
        # Run test.
        actual = scorer.summarize(rows)
        # Check outputs.
        self.assertAlmostEqual(actual["macro_f1"], 1.0)
        self.assertEqual(actual["n_scored"], 3)

    def test2(self) -> None:
        """
        Test classification with one incorrect prediction.
        """
        # Prepare inputs.
        scorer = tweweuti.MacroF1Scorer()
        rows = [
            {"true_label": 0, "predicted_label": 0},
            {"true_label": 1, "predicted_label": 0},
            {"true_label": 2, "predicted_label": 2},
        ]
        # Run test.
        actual = scorer.summarize(rows)
        # Check outputs.
        self.assertAlmostEqual(actual["f1_dovish"], 2.0 / 3.0)
        self.assertAlmostEqual(actual["f1_hawkish"], 0.0)
        self.assertAlmostEqual(actual["f1_neutral"], 1.0)
        self.assertAlmostEqual(actual["macro_f1"], 5.0 / 9.0)


# #############################################################################
# Test_align_fomc_yields
# #############################################################################


class Test_align_fomc_yields(hunitest.TestCase):
    """
    Test announcement and Treasury yield alignment.
    """

    def _make_treasury_data(self) -> pd.DataFrame:
        """
        Create synthetic Treasury yield observations.

        :return: daily Treasury yields
        """
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-04", "2024-01-05", "2024-01-08"]
                ),
                "yield_2yr": [4.00, 4.10, 4.05],
            }
        )

    def test1(self) -> None:
        """
        Test same-day alignment and basis-point conversion.
        """
        # Prepare inputs.
        statements = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-05"]),
                "statement_text": ["Example statement."],
                "source_url": ["https://example.com"],
            }
        )
        treasury = self._make_treasury_data()
        # Run test.
        actual = tweweuti.align_fomc_yields(statements, treasury)
        # Check outputs.
        self.assertEqual(len(actual), 1)
        self.assertTrue(actual["same_day_yield"].iloc[0])
        self.assertAlmostEqual(
            actual["yield_change_bps"].iloc[0], 10.0
        )

    def test2(self) -> None:
        """
        Test alignment of a Sunday announcement to Monday.
        """
        # Prepare inputs.
        statements = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-07"]),
                "statement_text": ["Example statement."],
                "source_url": ["https://example.com"],
            }
        )
        treasury = self._make_treasury_data()
        # Run test.
        actual = tweweuti.align_fomc_yields(statements, treasury)
        # Check outputs.
        self.assertFalse(actual["same_day_yield"].iloc[0])
        self.assertEqual(
            actual["yield_observation_date"].iloc[0],
            pd.Timestamp("2024-01-08"),
        )
        self.assertAlmostEqual(
            actual["yield_change_bps"].iloc[0], -5.0
        )

    def test3(self) -> None:
        """
        Test that both announcement dates are preserved.
        """
        # Prepare inputs.
        statements = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-05", "2024-01-07"]
                ),
                "statement_text": ["Statement one.", "Statement two."],
                "source_url": [
                    "https://example.com/1",
                    "https://example.com/2",
                ],
            }
        )
        treasury = self._make_treasury_data()
        # Run test.
        actual = tweweuti.align_fomc_yields(statements, treasury)
        # Check outputs.
        self.assertEqual(len(actual), 2)
        self.assertEqual(
            actual["date"].tolist(),
            statements["date"].tolist(),
        )


# #############################################################################
# Test_statement_processing
# #############################################################################


class Test_statement_processing(hunitest.TestCase):
    """
    Test statement cleaning and sentence splitting.
    """

    def test1(self) -> None:
        """
        Test removal of publication metadata.
        """
        # Prepare inputs.
        body = (
            "Economic conditions remain stable. "
            "The committee continues to monitor inflation. "
        ) * 6
        text = "January 5, 2024\nFor immediate release\nShare\n" + body
        # Run test.
        actual = tweweuti.clean_statement_text(text)
        # Check outputs.
        self.assertEqual(actual, body.strip())
        self.assertNotIn("For immediate release", actual)

    def test2(self) -> None:
        """
        Test that common abbreviations do not split sentences.
        """
        # Prepare inputs.
        sentence = (
            "The U.S. economy continues to expand at a moderate pace."
        )
        text = " ".join([sentence] * 6)
        # Run test.
        actual = tweweuti.split_sentences(text)
        # Check outputs.
        self.assertEqual(actual, [sentence] * 6)

    def test3(self) -> None:
        """
        Test sentence splitting across multiple lines.
        """
        # Prepare inputs.
        sentence1 = (
            "Economic activity continues to expand at a moderate pace."
        )
        sentence2 = (
            "The committee will monitor incoming economic information."
        )
        text = "\n".join([sentence1, sentence2] * 3)
        # Run test.
        actual = tweweuti.split_sentences(text)
        # Check outputs.
        self.assertEqual(actual, [sentence1, sentence2] * 3)


# #############################################################################
# Test_calculate_tone_correlations
# #############################################################################


class Test_calculate_tone_correlations(hunitest.TestCase):
    """
    Test Spearman rank correlations.
    """

    def _make_tones(self) -> pd.DataFrame:
        """
        Create synthetic positive and negative associations.

        :return: synthetic tone changes and yield changes
        """
        return pd.DataFrame(
            {
                "baseline_tone_change": [np.nan, -2, -1, 1, 2],
                "balanced_tone_change": [np.nan, 2, 1, -1, -2],
                "yield_change_bps": [0, -10, -5, 5, 10],
            }
        )

    def test1(self) -> None:
        """
        Test a perfect positive rank correlation.
        """
        # Prepare inputs.
        tones = self._make_tones()
        # Run test.
        actual = tweweuti.calculate_tone_correlations(
            tones, ["baseline"]
        )
        # Check outputs.
        self.assertAlmostEqual(
            actual.loc["baseline", "spearman_rho"], 1.0
        )

    def test2(self) -> None:
        """
        Test a perfect negative rank correlation.
        """
        # Prepare inputs.
        tones = self._make_tones()
        # Run test.
        actual = tweweuti.calculate_tone_correlations(
            tones, ["balanced"]
        )
        # Check outputs.
        self.assertAlmostEqual(
            actual.loc["balanced", "spearman_rho"], -1.0
        )

    def test3(self) -> None:
        """
        Test that missing tone changes are excluded.
        """
        # Prepare inputs.
        tones = self._make_tones()
        # Run test.
        actual = tweweuti.calculate_tone_correlations(
            tones, ["baseline", "balanced"]
        )
        # Check outputs.
        self.assertEqual(
            actual["n_observations"].tolist(), [4, 4]
        )


# #############################################################################
# Test_bootstrap_correlation_difference
# #############################################################################


class Test_bootstrap_correlation_difference(hunitest.TestCase):
    """
    Test paired bootstrap correlation comparisons.
    """

    def _make_tones(self) -> pd.DataFrame:
        """
        Create synthetic tone and yield changes.

        :return: synthetic announcement observations
        """
        return pd.DataFrame(
            {
                "baseline_tone_change": [-3, -2, -1, 1, 2, 3],
                "balanced_tone_change": [-2, -3, 1, -1, 3, 2],
                "yield_change_bps": [-10, -5, -2, 2, 5, 10],
            }
        )

    def test1(self) -> None:
        """
        Test reproducibility with a fixed random seed.
        """
        # Prepare inputs.
        tones = self._make_tones()
        # Run test.
        actual1 = tweweuti.bootstrap_correlation_difference(
            tones, "baseline", "balanced", 100, 42
        )
        actual2 = tweweuti.bootstrap_correlation_difference(
            tones, "baseline", "balanced", 100, 42
        )
        # Check outputs.
        self.assertEqual(actual1, actual2)

    def test2(self) -> None:
        """
        Test identical models produce a zero difference.
        """
        # Prepare inputs.
        tones = self._make_tones()
        tones["balanced_tone_change"] = tones[
            "baseline_tone_change"
        ]
        # Run test.
        actual = tweweuti.bootstrap_correlation_difference(
            tones, "baseline", "balanced", 100, 42
        )
        # Check outputs.
        self.assertAlmostEqual(actual["observed_difference"], 0.0)
        self.assertAlmostEqual(actual["ci_lower"], 0.0)
        self.assertAlmostEqual(actual["ci_upper"], 0.0)

    def test3(self) -> None:
        """
        Test valid confidence interval bounds.
        """
        # Prepare inputs.
        tones = self._make_tones()
        # Run test.
        actual = tweweuti.bootstrap_correlation_difference(
            tones, "baseline", "balanced", 100, 42
        )
        # Check outputs.
        self.assertLessEqual(
            actual["ci_lower"], actual["ci_upper"]
        )
        self.assertGreater(actual["n_valid_bootstrap"], 0)
        self.assertLessEqual(actual["n_valid_bootstrap"], 100)
        self.assertEqual(actual["n_observations"], 6)


# #############################################################################
# Test_plot_tone_correlations
# #############################################################################


class Test_plot_tone_correlations(hunitest.TestCase):
    """
    Test financial correlation visualization.
    """

    def test1(self) -> None:
        """
        Test that the figure contains one axis per model.
        """
        # Prepare inputs.
        tones = pd.DataFrame(
            {
                "baseline_tone_change": [-0.2, -0.1, 0.1, 0.2],
                "balanced_tone_change": [-0.1, -0.2, 0.2, 0.1],
                "yield_change_bps": [-10, -5, 5, 10],
            }
        )
        # Run test.
        fig = tweweuti.plot_tone_correlations(
            tones, ["baseline", "balanced"]
        )
        # Check outputs.
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(
                fig.axes[0].get_xlim(), fig.axes[1].get_xlim()
            )
            self.assertEqual(
                fig.axes[0].get_ylim(), fig.axes[1].get_ylim()
            )
        finally:
            plt.close(fig)

# #############################################################################
# _FixedPredictionModel
# #############################################################################


class _FixedPredictionModel:
    """
    Return predetermined labels for synthetic statement sentences.
    """

    def __init__(self, labels: list[int]) -> None:
        """
        Initialize the synthetic model.

        :param labels: predetermined classification labels
        """
        self._labels = labels

    def predict_batch(self, sentences: list[str]) -> list[int]:
        """
        Return predetermined predictions.

        :param sentences: sentences to classify
        :return: predicted labels
        """
        return self._labels[:len(sentences)]


# #############################################################################
# Test_score_statement_tone
# #############################################################################


class Test_score_statement_tone(hunitest.TestCase):
    """
    Test statement-level monetary policy tone scoring.
    """

    def _make_statement(self) -> str:
        """
        Create a synthetic statement containing six sentences.

        :return: synthetic statement text
        """
        sentence = (
            "The committee continues to monitor economic conditions."
        )
        return " ".join([sentence] * 6)

    def test1(self) -> None:
        """
        Test tone scoring for mixed classification labels.
        """
        # Prepare inputs.
        statement = self._make_statement()
        model = _FixedPredictionModel([0, 1, 1, 1, 2, 2])
        # Run test.
        actual = tweweuti.score_statement_tone(model, statement)
        # Check outputs.
        self.assertEqual(actual["n_sentences"], 6)
        self.assertEqual(actual["n_dovish"], 1)
        self.assertEqual(actual["n_hawkish"], 3)
        self.assertEqual(actual["n_neutral"], 2)
        self.assertAlmostEqual(actual["dovish_share"], 1.0 / 6.0)
        self.assertAlmostEqual(actual["hawkish_share"], 3.0 / 6.0)
        self.assertAlmostEqual(actual["neutral_share"], 2.0 / 6.0)
        self.assertAlmostEqual(actual["tone"], 2.0 / 6.0)

    def test2(self) -> None:
        """
        Test the extreme hawkish and dovish tone values.
        """
        # Prepare inputs.
        statement = self._make_statement()
        hawkish_model = _FixedPredictionModel([1] * 6)
        dovish_model = _FixedPredictionModel([0] * 6)
        # Run test.
        hawkish = tweweuti.score_statement_tone(
            hawkish_model, statement
        )
        dovish = tweweuti.score_statement_tone(
            dovish_model, statement
        )
        # Check outputs.
        self.assertAlmostEqual(hawkish["tone"], 1.0)
        self.assertAlmostEqual(dovish["tone"], -1.0)

    def test3(self) -> None:
        """
        Test that an entirely neutral statement has zero tone.
        """
        # Prepare inputs.
        statement = self._make_statement()
        model = _FixedPredictionModel([2] * 6)
        # Run test.
        actual = tweweuti.score_statement_tone(model, statement)
        # Check outputs.
        self.assertAlmostEqual(actual["tone"], 0.0)
        self.assertAlmostEqual(actual["neutral_share"], 1.0)


# #############################################################################
# Test_calculate_statement_tones
# #############################################################################


class Test_calculate_statement_tones(hunitest.TestCase):
    """
    Test statement tone aggregation and consecutive tone changes.
    """

    def _make_aligned_data(self) -> pd.DataFrame:
        """
        Create synthetic aligned statement and yield observations.

        :return: aligned announcement data
        """
        sentence = (
            "The committee continues to monitor economic conditions."
        )
        statement = " ".join([sentence] * 6)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-31", "2024-03-20", "2024-05-01"]
                ),
                "statement_text": [statement] * 3,
                "yield_change_bps": [5.0, -3.0, 2.0],
            }
        )

    def test1(self) -> None:
        """
        Test tone aggregation for two classification models.
        """
        # Prepare inputs.
        aligned = self._make_aligned_data()
        models = {
            "baseline": _FixedPredictionModel([1] * 6),
            "balanced": _FixedPredictionModel([0] * 6),
        }
        # Run test.
        actual = tweweuti.calculate_statement_tones(
            aligned, models
        )
        # Check outputs.
        self.assertEqual(len(actual), 3)
        self.assertEqual(
            actual["baseline_tone"].tolist(),
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            actual["balanced_tone"].tolist(),
            [-1.0, -1.0, -1.0],
        )
        self.assertEqual(
            actual["yield_change_bps"].tolist(),
            [5.0, -3.0, 2.0],
        )

    def test2(self) -> None:
        """
        Test chronological ordering and consecutive tone changes.
        """
        # Prepare inputs.
        aligned = self._make_aligned_data()

        class _ChangingPredictionModel:
            """
            Return different predictions for consecutive statements.
            """

            def predict_batch(
                self, sentences: list[str]
            ) -> list[int]:
                """
                Classify sentences using synthetic keywords.

                :param sentences: sentences to classify
                :return: predicted labels
                """
                return [
                    0 if "lower" in sentence.lower()
                    else 1 if "raise" in sentence.lower()
                    else 2
                    for sentence in sentences
                ]

        aligned["statement_text"] = [
            "The committee will lower rates to support the economy. " * 6,
            "The committee will maintain its current policy stance. " * 6,
            "The committee will raise rates to address inflation. " * 6,
        ]
        aligned = aligned.iloc[::-1].reset_index(drop=True)
        models = {"baseline": _ChangingPredictionModel()}
        # Run test.
        actual = tweweuti.calculate_statement_tones(
            aligned, models
        )
        # Check outputs.
        self.assertTrue(actual["date"].is_monotonic_increasing)
        self.assertEqual(
            actual["baseline_tone"].tolist(),
            [-1.0, 0.0, 1.0],
        )
        changes = actual["baseline_tone_change"]
        self.assertTrue(pd.isna(changes.iloc[0]))
        self.assertEqual(changes.iloc[1:].tolist(), [1.0, 1.0])