# Reproduce the recorded analysis without paid calls

`evidence/chess_m2_completed_20261119.tar.gz` preserves the source, pinned
dependency lists, complete first-stage and remaining-stage working stores,
transactional spending ledgers, execution manifests, reports and plots.
It also includes the original M1 archive and local Syzygy tablebase files.
The API credential and virtual environment are excluded. The sibling
`.sha256` file checks the final archive; `final_artifact_inventory.json`
records individual payload file hashes.

From an extracted copy of the project, with Python 3.12 and `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-analysis.lock
mkdir -p inputs/m1_original
tar -xzf inputs/archive/m1_runs.tar.gz -C inputs/m1_original
.venv/bin/python scripts/report_remaining_m2.py
.venv/bin/python -m pytest -q -c /dev/null tests vendor/pr917/test/test_caaf.py
```

The reporting command performs no model inference. It checks all 3,900
completed samples, including the 16 continued across administrative branch
boundaries, and regenerates the CSV/JSON analysis and PNG/PDF figures. Tests
use recorded or mocked replies and make no paid calls. Extract the source
archive only into a dedicated project copy; the live original inputs were
preserved unchanged during the experiment.

Expected results: 16,165 settled paid model calls in the remaining phases,
$14.075944970 usage-rated cost, 18,115 replayed decision steps, 3,916 stored
branch segments for 3,900 samples, and three harmful primary decisions after
Holm correction across 48 tests. Including the first validation stage, total
usage-rated cost is $15.603057070. No outstanding/uncertain request remains.

`execution_manifest.json` preserves the initial runtime hashes and
`resume_epoch_01/execution_manifest.json` preserves the resumed runtime
hashes. Only the old, mistakenly included pytest cache changed; pinned
executable and input hashes match. Analysis source hashes are in `REPORT.json`.
The original M1 archive SHA-256 is
`d63ef2d94a695280cffa1896bbe201e0f2d3f79def9b90751801c6f09e3a7172`.

This reproduces recorded environment transitions and the analysis, not fresh
model replies. The hosted model has no seed in this protocol, and no provider
weight digest is recorded. Independent repeats are statistical evidence;
they do not prove equality of every effect. The original M1 checkout had a
dirty-source limitation that remains documented in its provenance.

The paid runners are retained as evidence. Collection is complete; do not
restart them under the completed manifest. Milestone 3/4 paid work requires
its own declared experiment and authorization.
