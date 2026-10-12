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
| `checker_canaries.py`   | Canary-bank validation and secret balanced selection     | Canaries   |
| `checker_evidence.py`   | Seller-attributable evidence and exclusion rules         | Evidence   |
| `checker_grading.py`    | Versioned deterministic answer normalization and grading | Canaries   |
| `checker_models.py`     | Immutable inputs and outputs shared by the checker       | Models     |
| `checker_statistics.py` | Wilson and exact one-sided upper bounds                  | Statistics |
| `checker_store.py`      | PostgreSQL reads, probe claims, and atomic finalization  | Database   |
| `checker_verdicts.py`   | Evidence minima, failure precedence, and verdicts        | Verdicts   |

## Description of Architecture

- `checker/` imports `common/` (schema in `common/migrations/002_checker.sql`) and
  the status constants of `gateway/gateway_providers.py`
- `checker_models.py` is the base: every other module imports it, and it imports
  nothing from the checker
