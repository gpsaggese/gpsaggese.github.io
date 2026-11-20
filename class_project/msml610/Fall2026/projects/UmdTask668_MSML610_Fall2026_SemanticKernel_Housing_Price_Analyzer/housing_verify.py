"""
Check the labeled metrics in an agent answer against the function results.

The check compares the numbers that the answer attaches to a known label
with the values returned by the executed functions:

- dataset facts from `dataset_summary`: number of rows, number of features,
  and missing-value count
- model metrics from `compare_models`: MAE, RMSE, and R² of each model

A reported value matches when it equals the function value after rounding
to the number of decimals used in the answer (e.g., `0.81` matches
`0.8083`); R² may also be reported as a percentage. Values are attributed to
a model when the sentence, line, or table row mentions exactly one model;
otherwise they are checked against the values of all models.

Limitations: the check only covers the labels above and the phrasings it
recognizes. Matching values do not show that the explanation, comparisons,
or conclusions in the answer are correct, and unrecognized phrasings are
reported as "not reported" rather than checked.

Import as:

import housing_verify
"""

import dataclasses
import json
import re

import housing_trace

DATASET_FUNCTION = "dataset_summary"
MODELS_FUNCTION = "compare_models"
METRICS = ("MAE", "RMSE", "R2")
METRIC_DISPLAY = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R²"}

# Metric labels recognized in answers.
_METRIC_PATTERNS = {
    "RMSE": r"\bRMSE\b|root[- ]mean[- ]squared[- ]error",
    "MAE": r"\bMAE\b|mean[- ]absolute[- ]error",
    "R2": r"R²|R\^2|\bR2\b|R-squared|\bR squared\b|coefficient of determination",
}
_METRIC_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _METRIC_PATTERNS.items()),
    re.IGNORECASE,
)
# A number that is not part of a word such as `X1` or `R2`.
_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(\s*%)?(?![\w])")
# Maximum distance between a metric label and its value, in characters.
_MAX_LABEL_DISTANCE = 40

_ROWS_PATTERNS = (
    r"(\d[\d,]*)\s+(?:rows|records|observations|samples|transactions)\b",
    r"\b(?:rows|records|observations|samples)\s*[:=]\s*(\d[\d,]*)",
)
_FEATURES_PATTERNS = (
    r"(\d+)\s+(?:input\s+|numeric\s+)?(?:features|predictors)\b",
    r"\b(?:number of features|features)\s*[:=]\s*(\d+)\b",
)
_MISSING_PATTERNS = (
    r"(\d+)\s+missing\b",
    r"\bmissing(?:[- ]values?)?\s*[:=]\s*(\d+)",
)
_NO_MISSING_RE = re.compile(r"\bno missing values?\b", re.IGNORECASE)


@dataclasses.dataclass
class MetricCheck:
    """
    Result of checking one expected value.

    :param metric: name of the checked value, e.g., `rows` or
        `Random Forest MAE`
    :param expected: value returned by the function
    :param reported: values the answer reports for this metric
    :param status: `verified` (all reported values match), `mismatch` (a
        reported value differs), or `not_reported`
    """

    metric: str
    expected: float
    reported: list[str] = dataclasses.field(default_factory=list)
    status: str = "not_reported"


@dataclasses.dataclass
class MetricCheckReport:
    """
    Result of checking an answer.

    :param checks: one check per expected value
    :param issues: reported values that contradict the function results or
        have no supporting function result
    """

    checks: list[MetricCheck]
    issues: list[str]

    @property
    def ok(self) -> bool:
        """Return True if no reported value contradicts the results."""
        return not self.issues and all(
            check.status != "mismatch" for check in self.checks
        )

    def format(self) -> str:
        """Return a human-readable summary of the check."""
        if not self.checks and not self.issues:
            return "No labeled metrics to check."
        lines = []
        for check in self.checks:
            reported = ", ".join(check.reported) if check.reported else "-"
            lines.append(
                f"{check.status:12s} {check.metric}: expected"
                f" {check.expected:g}, reported {reported}"
            )
        lines.extend(f"issue        {issue}" for issue in self.issues)
        return "\n".join(lines)


def _matches(reported: str, is_percent: bool, expected: float) -> bool:
    """Return True if `reported` equals `expected` up to its rounding."""
    value = float(reported.replace(",", ""))
    decimals = len(reported.split(".")[1]) if "." in reported else 0
    if is_percent:
        value /= 100
        decimals += 2
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return abs(value - expected) <= tolerance


def _latest_result(trace: housing_trace.ExecutionTrace, function: str):
    """Return the parsed result of the last successful call of a function."""
    for record in reversed(trace.records):
        if record.function == function and record.status == "ok":
            try:
                return json.loads(record.result)
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def _model_patterns(models: list[str]) -> dict[str, re.Pattern]:
    patterns = {}
    for model in models:
        initials = "".join(word[0] for word in model.split())
        patterns[model] = re.compile(
            rf"{re.escape(model)}|\b{re.escape(initials)}\b", re.IGNORECASE
        )
    return patterns


def _mentioned_models(text: str, patterns: dict[str, re.Pattern]) -> list[str]:
    return [model for model, pattern in patterns.items() if pattern.search(text)]


def _segments(text: str) -> list[str]:
    """Split non-table text into lines and sentences."""
    segments = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            continue
        segments.extend(
            part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if part
        )
    return segments


def _table_values(text: str, patterns: dict[str, re.Pattern]):
    """
    Yield (model, metric, number, is_percent) from markdown tables.

    Supports tables with models as rows and metrics as columns, and tables
    with metrics as rows and models as columns.
    """
    # Group consecutive table lines into separate tables.
    tables: list[list[list[str]]] = []
    previous_was_table = False
    for line in text.splitlines():
        is_table = line.strip().startswith("|")
        if is_table:
            if not previous_was_table:
                tables.append([])
            tables[-1].append(
                [cell.strip() for cell in line.strip().strip("|").split("|")]
            )
        previous_was_table = is_table
    for table in tables:
        rows = [
            cells for cells in table
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c)
        ]
        if len(rows) < 2:
            continue
        header, body = rows[0], rows[1:]
        header_metrics = [_metric_of(cell) for cell in header]
        header_models = [
            (_mentioned_models(cell, patterns) or [None])[0] for cell in header
        ]
        for cells in body:
            row_models = _mentioned_models(cells[0], patterns)
            row_metric = _metric_of(cells[0])
            for index, cell in enumerate(cells[1:], start=1):
                match = _NUMBER_RE.search(cell)
                if match is None or index >= len(header):
                    continue
                number, is_percent = match.group(1), bool(match.group(2))
                if header_metrics[index] and len(row_models) == 1:
                    yield row_models[0], header_metrics[index], number, is_percent
                elif row_metric and header_models[index]:
                    yield header_models[index], row_metric, number, is_percent


def _metric_of(text: str) -> str | None:
    match = _METRIC_RE.search(text)
    return match.lastgroup if match else None


def _text_values(text: str, patterns: dict[str, re.Pattern]):
    """Yield (model or None, metric, number, is_percent) from sentences."""
    for segment in _segments(text):
        labels = list(_METRIC_RE.finditer(segment))
        if not labels:
            continue
        models = _mentioned_models(segment, patterns)
        model = models[0] if len(models) == 1 else None
        for number in _NUMBER_RE.finditer(segment):
            preceding = [m for m in labels if m.end() <= number.start()]
            if not preceding:
                continue
            label = preceding[-1]
            if number.start() - label.end() > _MAX_LABEL_DISTANCE:
                continue
            yield model, label.lastgroup, number.group(1), bool(number.group(2))


def _find_counts(text: str, patterns: tuple[str, ...]) -> list[str]:
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, re.IGNORECASE))
    return values


def _check_count(name: str, expected: float, reported: list[str]) -> MetricCheck:
    check = MetricCheck(metric=name, expected=expected, reported=reported)
    if reported:
        check.status = (
            "verified"
            if all(_matches(value, False, expected) for value in reported)
            else "mismatch"
        )
    return check


def check_reported_metrics(
    answer: str, trace: housing_trace.ExecutionTrace
) -> MetricCheckReport:
    """
    Compare the labeled metrics in an answer with the function results.

    :param answer: final answer of the agent
    :param trace: execution trace of the run that produced the answer
    :return: one check per expected value, plus issues for reported values
        that contradict the results or have no supporting result
    """
    # Remove markdown emphasis so that `**414 rows**` reads as `414 rows`.
    text = re.sub(r"[*_`]", "", answer)
    checks: list[MetricCheck] = []
    issues: list[str] = []
    summary = _latest_result(trace, DATASET_FUNCTION)
    if isinstance(summary, dict):
        if "rows" in summary:
            checks.append(_check_count(
                "rows", summary["rows"], _find_counts(text, _ROWS_PATTERNS)))
        if isinstance(summary.get("features"), list):
            checks.append(_check_count(
                "features", len(summary["features"]),
                _find_counts(text, _FEATURES_PATTERNS)))
        if "missing_values" in summary:
            missing = _find_counts(text, _MISSING_PATTERNS)
            if _NO_MISSING_RE.search(text):
                missing.append("0")
            checks.append(_check_count(
                "missing_values", summary["missing_values"], missing))
    scores = _latest_result(trace, MODELS_FUNCTION)
    if not isinstance(scores, dict):
        scores = {}
    patterns = _model_patterns(list(scores))
    model_checks = {
        (model, metric): MetricCheck(
            metric=f"{model} {METRIC_DISPLAY[metric]}",
            expected=float(values[metric]),
        )
        for model, values in scores.items()
        for metric in METRICS
        if metric in values
    }
    found = list(_table_values(text, patterns)) + list(_text_values(text, patterns))
    for model, metric, number, is_percent in found:
        label = f"{number}{'%' if is_percent else ''}"
        candidates = {
            key: check for key, check in model_checks.items()
            if key[1] == metric and (model is None or key[0] == model)
        }
        if not candidates:
            issues.append(
                f"{METRIC_DISPLAY[metric]} {label} is reported, but no"
                f" {MODELS_FUNCTION} result supports it"
            )
            continue
        matching = [
            check for check in candidates.values()
            if _matches(number, is_percent, check.expected)
        ]
        if matching:
            for check in matching:
                check.reported.append(label)
                if check.status == "not_reported":
                    check.status = "verified"
        elif model is not None:
            check = candidates[(model, metric)]
            check.reported.append(label)
            check.status = "mismatch"
        else:
            issues.append(
                f"{METRIC_DISPLAY[metric]} {label} matches no model's"
                f" {METRIC_DISPLAY[metric]}"
            )
    checks.extend(model_checks.values())
    return MetricCheckReport(checks=checks, issues=issues)
