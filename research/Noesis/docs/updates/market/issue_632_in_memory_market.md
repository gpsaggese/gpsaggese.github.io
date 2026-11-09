# Issue 632: In-Memory Market Demonstration

## Summary

- PR_M13 demonstrates all five named M11 scenarios through the M10 round runner.
- The notebook reuses M12 tables and plots, with paired Python source and reusable
  utilities; fulfillment is explicitly mocked.

## Architecture

```text
Seeded scenario -> Input orders -> In-memory book and contract store
                                           |
                                           v
                               run_round() with mocked fulfillment
                                           |
                                           v
                               M12 tables and per-tier curves
```

## Protocol

- Use seed `7` by default; change it to explore another reproducible example.
- Each scenario creates a fresh book and store; round IDs restart at `0`, and
  every tier in that round shares the same ID.
- Display submitted orders, tier summaries, contracts, unmatched quantities, and
  supply/demand curves; the fulfillment callback always returns `True`.

## Primary Results

| Scenario | Observed Outcome |
| --- | --- |
| Balanced | All demand and supply fill |
| Excess demand | Buyer quantity remains unmatched |
| Excess supply | Seller quantity remains unmatched |
| Multiple tiers | Tiers clear at distinct prices within one round |
| Non-crossing prices | No trades, contracts, or clearing price |

- Headless execution produces HTML with 20 tables and six plots, including when
  Python socket connections are rejected.
- Checks confirm volume conservation and unchanged input orders; all 711 existing
  market, report-demo, and component-boundary tests pass.
- Paired Python execution and repository formatting and pairing checks pass.

## Claims and Limitations

- Mocked fulfillment does not demonstrate real delivery, payments, or provider
  performance; the notebook uses no database or HTTP calls.
- Validation uses local `nbconvert`; the plan's Docker-based runner is unverified.
  This PR adds no Docker or GitHub workflow configuration.
- The local kernel reports a sandbox permission error during shutdown after all
  cells finish; HTML export still succeeds.

## Files

- `notebooks/market_in_memory.ipynb`: Scenario walkthrough with tables and curves.
- `notebooks/market_in_memory.py`: Jupytext-paired source.
- `notebooks/market_in_memory_utils.py`: Submission and presentation helpers.
- [Notebook README](../../../notebooks/README.md): Local execution instructions.
