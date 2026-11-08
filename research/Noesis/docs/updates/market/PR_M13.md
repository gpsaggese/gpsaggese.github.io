# PR_M13: In-Memory Market Demonstration

- **Behavior**: demonstrates all five M11 scenario families through M10, with M12
  input, summary, contract, unmatched-order tables, and a curve for every tier
- **Decisions**:
  - Each scenario uses a fresh in-memory book and contract store for independent
    reruns; round IDs restart at 0, and multi-tier results share one round ID
  - Fulfillment always returns `True` and is explicitly labeled mocked
  - Notebook cells show inputs and calls; utilities own presentation and submission
- **Review**: see [notebook execution instructions](../../../notebooks/README.md)
- **Validation**:
  - Use the local `nbconvert` command in the review instructions; the plan's
    Docker-based `run_nbconvert.py` path is unverified, with no Docker or GitHub
    workflow additions in this PR as requested
  - Headless execution produces HTML, 20 tables, and six plots; a second execution
    succeeds with Python socket connections rejected
  - All 711 existing market, report-demo, and component-boundary tests pass
  - The local kernel reports a sandbox permission error while enumerating child
    processes at shutdown; all notebook cells finish and HTML is produced
