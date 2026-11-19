"""Route synthetic support tickets with d1-3B and an explicit cost policy.

Run from the repository root while the decision server is listening:

> python research/Implement_System_One_Decision_Models/ticket_routing.py \
    --split dev --output /tmp/dev_predictions.json
"""

import argparse
import csv
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import research.Implement_System_One_Decision_Models.system_one_API_utils as soau

# #############################################################################
# Frozen application configuration
# #############################################################################


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.csv")
ROUTES = ("billing", "shipping", "technical")
ROUTE_QUESTION = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this customer request?",
        "criteria": {
            "billing": "Payments and refunds",
            "shipping": "Delivery and tracking",
            "technical": "Product or account malfunction",
        },
    }
}
# Rows are actions; columns are the customer's true route. The units are
# illustrative delay/handling costs, chosen before evaluating the test split.
ACTION_COSTS = {
    "billing": {"billing": 0.0, "shipping": 5.0, "technical": 7.0},
    "shipping": {"billing": 8.0, "shipping": 0.0, "technical": 6.0},
    "technical": {"billing": 8.0, "shipping": 6.0, "technical": 0.0},
    "review": {"billing": 1.5, "shipping": 1.5, "technical": 1.5},
}
MODEL_REPOSITORY = "LiquidAI/d1-3B-GGUF"
MODEL_REVISION = "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563"
MODEL_FILE = "d1-3B-Q8_0.gguf"
MODEL_SHA256 = "2f0942d5a5f64cf69c3356d2be439b71644b1f0eb3a580912a5a0179eeaba77d"
LLAMA_CPP_COMMIT = "8956c769e8ababd9fbd89c81764f241d5f515d68"
PROMPT_VERSION = "support-route-v1"


# #############################################################################
# Data and policy
# #############################################################################


def load_cases(split: str, data_path: str = DATA_PATH) -> List[Dict[str, str]]:
    """Load the fixed split and reject duplicate IDs or unknown labels."""
    if split not in ("dev", "test"):
        raise ValueError("Split must be 'dev' or 'test'")
    with open(data_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Ticket IDs must be unique")
    if any(row["split"] not in ("dev", "test") for row in rows):
        raise ValueError("Unexpected split in ticket data")
    if any(row["label"] not in ROUTES or not row["text"] for row in rows):
        raise ValueError("Unexpected label or empty ticket text")
    return [row for row in rows if row["split"] == split]


def expected_costs(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Compute each action's expected cost under the model distribution."""
    if set(probabilities) != set(ROUTES):
        raise ValueError("The distribution must cover exactly the three routes")
    if any(value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("Probabilities must lie in [0, 1]")
    if abs(sum(probabilities.values()) - 1.0) > 1e-3:
        raise ValueError("Probabilities must sum to one")
    return {
        action: sum(
            probabilities[true_route] * costs[true_route]
            for true_route in ROUTES
        )
        for action, costs in ACTION_COSTS.items()
    }


def choose_min_cost_action(probabilities: Dict[str, float]) -> str:
    """Choose a route or review by the smallest model-expected cost."""
    costs = expected_costs(probabilities)
    return min(costs, key=costs.get)


def actual_cost(action: str, true_route: str) -> float:
    """Return the illustrative realized cost of one action and true route."""
    return ACTION_COSTS[action][true_route]


# #############################################################################
# Inference and command line
# #############################################################################


def run_split(
    split: str, *, url: Optional[str] = None, data_path: str = DATA_PATH
) -> Dict[str, Any]:
    """Predict one fixed split using the frozen prompt and policy."""
    client = soau.SystemOneClient(url)
    cases = load_cases(split, data_path)
    with open(data_path, "rb") as stream:
        data_sha256 = hashlib.sha256(stream.read()).hexdigest()
    records = []
    for index, case in enumerate(cases, start=1):
        started = time.monotonic()
        result = client.system_one(case["text"], ROUTE_QUESTION)
        seconds = time.monotonic() - started
        answer = result["answers"]["route"]
        probabilities = soau.choice_probabilities(answer, list(ROUTES))
        top_route = soau.choose_top_option(probabilities)
        policy_action = choose_min_cost_action(probabilities)
        records.append(
            {
                "id": case["id"],
                "true_route": case["label"],
                "text": case["text"],
                "probabilities": probabilities,
                "top_route": top_route,
                "policy_action": policy_action,
                "expected_costs": expected_costs(probabilities),
                "seconds": round(seconds, 3),
            }
        )
        print(
            f"{split}: {index}/{len(cases)} {case['id']} -> {policy_action}",
            flush=True,
        )
    return {
        "split": split,
        "dataset_sha256": data_sha256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_file": MODEL_FILE,
        "model_sha256": MODEL_SHA256,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "prompt_version": PROMPT_VERSION,
        "question": ROUTE_QUESTION,
        "action_costs": ACTION_COSTS,
        "records": records,
    }


def main() -> None:
    """Run one split and write all predictions for later evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    report = run_split(args.split, url=args.url)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {len(report['records'])} cases to {args.output}")


if __name__ == "__main__":
    main()
