# Checker

- Answers one question: did a seller deliver the latency and quality a contract
  promised?
- Turns immutable request records into a `PASSED`, `FAILED`, or `INSUFFICIENT`
  verdict, with the evidence behind it
- Decision logic is pure; database access sits in `checker_store.py`
- Design, evaluation flow, and window policy: `docs/checker_design.md`

## Structure of the Dir

| Dir     | Description                            |
| :------ | :------------------------------------- |
| `test/` | Unit tests, one `test_*.py` per module |

## Description of Files

| File                    | Description                                              | Cluster    |
| :---------------------- | :------------------------------------------------------- | :--------- |
| `checker_calibration.py` | Pure acceptance decisions for calibration observations  | Canaries   |
| `checker_calibration_report.py` | Markdown findings from one completed calibration | Canaries |
| `checker_calibration_runtime.py` | Concurrent provider calls and persisted calibration | Canaries |
| `checker_capability.py` | Pluggable measurement protocol and exact-match implementation | Canaries |
| `checker_canaries.py`   | Canary-bank validation and secret balanced selection     | Canaries   |
| `checker_evidence.py`   | Seller-attributable evidence and exclusion rules         | Evidence   |
| `checker_grading.py`    | Versioned deterministic answer normalization and grading | Canaries   |
| `checker_models.py`     | Immutable inputs and outputs shared by the checker       | Models     |
| `checker_probe_schedule.py` | Secret, balanced probe timing across a contract window | Canaries |
| `checker_statistics.py` | Wilson and exact one-sided upper bounds                  | Statistics |
| `checker_store.py`      | Calibration, probe, evidence, and verdict persistence    | Database   |
| `checker_verdicts.py`   | Evidence minima, failure precedence, and verdicts        | Verdicts   |

## Description of Architecture

- `checker/` imports `common/` (schema in `common/migrations/002_checker.sql`) and
  the status constants of `gateway/gateway_providers.py`
- `checker_models.py` is the base: every other module imports it, and it imports
  nothing from the checker
