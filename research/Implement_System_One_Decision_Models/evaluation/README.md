# Ticket Routing Evaluation

## Fixed Setup

The application configuration was committed in Phase 3 before the test split
was run. The dataset has 36 development and 36 held-out test cases, with 12
of each route in each split. All cases are original synthetic text. The
dataset SHA-256 is
`ed073d72c5cdda159fff9541dbbc639b3820d15904c93e627e4488f81e47177b`.

The run used prompt version `support-route-v1`, the Q8 d1-3B model file at
revision `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`, and llama.cpp
commit `8956c769e8ababd9fbd89c81764f241d5f515d68`. See
[`ticket_routing.py`](../ticket_routing.py) for the exact question, route
descriptions, and cost table. The raw per-ticket probabilities are in
[`test_predictions.json`](../test_predictions.json). No test labels were used
to change the prompt or policy.

## Test Results

| Measure | Result | Definition |
| :--- | ---: | :--- |
| Top-route accuracy | 33/36 = 91.7% | Highest-probability route equals the hand label |
| Majority-route accuracy | 12/36 = 33.3% | Always choose `billing`, the deterministic tie choice on balanced development labels |
| Multiclass Brier score | 0.0973 | Mean sum of three squared probability errors; lower is better |
| Negative log likelihood | 0.1793 | Mean negative natural log of the hand-label probability; lower is better |
| Five-bin top-label ECE | 0.0488 | Count-weighted gap between bin accuracy and mean top probability |
| Median request time | 1.370 seconds | One run's per-request elapsed times; descriptive only |

The reliability bins contain 3, 4, and 29 cases in the occupied ranges. The
other two bins are empty. The [reliability plot](reliability.png) makes the
counts visible. With so few cases, the ECE number and the apparent closeness
to the diagonal are unstable; they do not establish calibration.

The top-route confusion table has true routes as rows and predicted routes
as columns:

| True route | Billing | Shipping | Technical |
| :--- | ---: | ---: | ---: |
| Billing | 12 | 0 | 0 |
| Shipping | 1 | 11 | 0 |
| Technical | 2 | 0 | 10 |

## From Probabilities to Actions

The fixed expected-cost policy compared all three routes with human review
at a cost of 1.5 units. It reviewed 6/36 cases, automatically routed 30/36
(83.3% coverage), and made 1 error among those 30 (3.3%). Its total realized
cost on these labels was 16 units, or 0.444 per case. Always taking the
top-probability route cost 19 units, or 0.528 per case. These units are
illustrative handling costs, not measured operational dollars. The
[coverage plot](coverage.png) orders cases by model-expected route cost and
marks the fixed policy's selected coverage.

Three top-route errors show what the aggregate numbers hide:

- `S-T-07`: A return-to-sender delivery case was given 0.511 for billing and
  0.402 for shipping. The cost policy sent it to review.
- `T-T-09`: A frozen order page before payment was given 0.863 for billing and
  0.126 for technical. The cost policy still routed it to billing, an error.
- `T-T-11`: A billing page crash during invoice download was given 0.501 for
  billing and 0.485 for technical. The cost policy sent it to review.

## Reproduce

Start the pinned server as described in the project README, then run from the
repository root:

```bash
> python research/Implement_System_One_Decision_Models/ticket_routing.py \
    --split dev \
    --output research/Implement_System_One_Decision_Models/dev_predictions.json
> python research/Implement_System_One_Decision_Models/ticket_routing.py \
    --split test \
    --output research/Implement_System_One_Decision_Models/test_predictions.json
> python research/Implement_System_One_Decision_Models/evaluate_routing.py
```

The evaluator verifies dataset and model hashes, question text, cost table,
case IDs, labels, and saved actions before computing metrics. It writes
`results.json`, `reliability.png`, and `coverage.png` in this directory.

## Limits

The cases are synthetic, small, and balanced by design. They may resemble
examples in the model's undisclosed training data; overlap cannot be ruled
out. Their hand labels express one primary route for mixed-topic tickets,
which can be debatable. The Q8 file can yield different probabilities from
full precision. The measured accuracy, ECE, costs, and latency apply to this
single frozen exercise; no population-level or production claim follows.
