"""Probe the typed decision API of a running d1-3B server.

Run with:

> python probe_decision_api.py --output /tmp/d1_phase1_probe.json
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

# #############################################################################
# Constants
# #############################################################################


DEFAULT_URL = "http://127.0.0.1:18089/v1/systemone"


# #############################################################################
# API and validation
# #############################################################################


def _request(url: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Post one state and its named questions; return the response and seconds."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as connection:
            response = json.load(connection)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Decision API returned HTTP {error.code}: {detail}"
        ) from error
    return response, time.monotonic() - started


def _probability(value: Any, label: str) -> float:
    """Require a finite probability in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric: {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{label} is outside [0, 1]: {value!r}")
    return probability


def _distribution(value: Any, keys: List[str], label: str) -> Dict[str, float]:
    """Check option identities, probability bounds, and normalization."""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} must have exactly these keys: {keys}")
    probabilities = {
        key: _probability(value[key], f"{label}.{key}") for key in keys
    }
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
        raise ValueError(f"{label} does not sum to one: {probabilities}")
    return probabilities


def _validate(response: Dict[str, Any], questions: Dict[str, Any]) -> None:
    """Check the response contract against the request's declared schema."""
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError(
            "Response answer names differ from request question names"
        )
    for name, question in questions.items():
        answer = answers[name]
        if (
            not isinstance(answer, dict)
            or answer.get("type") != question["type"]
        ):
            raise ValueError(f"{name} has the wrong answer type")
        if question["type"] == "noul":
            _probability(answer.get("noul"), f"{name}.noul")
            continue
        if question["type"] == "choice":
            keys = list(question["criteria"])
            probabilities = _distribution(
                answer.get("probabilities"), keys, f"{name}.probabilities"
            )
            choice = answer.get("choice")
            if (
                choice not in keys
                or probabilities[choice] < max(probabilities.values()) - 1e-6
            ):
                raise ValueError(
                    f"{name}.choice does not match the probabilities"
                )
        elif question["type"] == "score":
            keys = [str(index) for index in range(len(question["criteria"]))]
            probabilities = _distribution(
                answer.get("probabilities"), keys, f"{name}.probabilities"
            )
            legend = answer.get("legend")
            expected_legend = dict(zip(keys, question["criteria"]))
            if legend != expected_legend:
                raise ValueError(
                    f"{name}.legend does not match the level descriptions"
                )
            score = answer.get("score")
            expected_score = sum(
                int(index) * probability
                for index, probability in probabilities.items()
            )
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{name}.score must be numeric")
            if not math.isclose(score, expected_score, abs_tol=1e-3):
                raise ValueError(f"{name}.score is not the expected level index")
        else:
            raise ValueError(f"Unknown question type: {question['type']}")
        _probability(answer.get("confidence"), f"{name}.confidence")


# #############################################################################
# Probe cases
# #############################################################################


def _base_questions() -> Dict[str, Any]:
    """Return three independent questions over a support request."""
    return {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this customer request?",
            "criteria": {
                "billing": "Payments and refunds",
                "shipping": "Delivery and tracking",
                "technical": "Product or account malfunction",
            },
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer explicitly request a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does this need a reply?",
            "criteria": ["can wait", "this week", "today", "immediately"],
        },
    }


def _run_case(
    url: str, name: str, state: Any, questions: Dict[str, Any]
) -> Dict[str, Any]:
    """Run one case and retain its request and validated response."""
    payload = {"state": state, "questions": questions}
    response, seconds = _request(url, payload)
    _validate(response, questions)
    return {
        "name": name,
        "seconds": round(seconds, 3),
        "request": payload,
        "response": response,
    }


def _max_probability_change(
    first: Dict[str, Any], second: Dict[str, Any]
) -> float:
    """Compare the probabilities of equivalent choice questions."""
    left = first["response"]["answers"]["route"]["probabilities"]
    right = second["response"]["answers"]["route"]["probabilities"]
    return max(abs(left[key] - right[key]) for key in left)


def run_probe(url: str) -> Dict[str, Any]:
    """Exercise text, JSON, all three types, new schema, and call behavior."""
    state = "Customer: My card was charged twice. Please refund the second charge today."
    questions = _base_questions()
    text_multi = _run_case(url, "text_multi", state, questions)
    json_multi = _run_case(
        url,
        "json_multi",
        {"channel": "email", "message": state, "priority": "standard"},
        questions,
    )
    unseen_choice = _run_case(
        url,
        "unseen_choice_schema",
        state,
        {
            "next_step": {
                "type": "choice",
                "instructions": "What should the support agent do next?",
                "criteria": {
                    "investigate_charge": "Check payment records",
                    "request_tracking": "Ask for a parcel tracking number",
                    "reset_password": "Send account reset instructions",
                    "human_review": "Ask a human specialist to decide",
                },
            }
        },
    )
    reversed_question = _base_questions()["route"]
    reversed_question["criteria"] = dict(
        reversed(list(reversed_question["criteria"].items()))
    )
    reordered_questions = _base_questions()
    reordered_questions["route"] = reversed_question
    reordered = _run_case(url, "reordered_choice", state, reordered_questions)
    repeated = _run_case(url, "repeated_text_multi", state, questions)
    single = _run_case(
        url, "single_choice", state, {"route": questions["route"]}
    )
    cases = [text_multi, json_multi, unseen_choice, reordered, repeated, single]
    return {
        "endpoint": url,
        "cases": cases,
        "observations": {
            "multi_question_count": len(questions),
            "option_order_max_probability_change": _max_probability_change(
                text_multi, reordered
            ),
            "repeat_identical_answers": (
                text_multi["response"]["answers"]
                == repeated["response"]["answers"]
            ),
            "single_vs_multi_route_identical": (
                text_multi["response"]["answers"]["route"]
                == single["response"]["answers"]["route"]
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    """Parse the server endpoint and optional report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the probe and print a concise, machine-readable summary."""
    args = _parse_args()
    report = run_probe(args.url)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    summary = {
        "case_names": [case["name"] for case in report["cases"]],
        "observations": report["observations"],
        "output": args.output,
        "validated": True,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
