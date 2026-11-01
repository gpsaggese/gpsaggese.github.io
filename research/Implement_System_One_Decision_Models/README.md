# Implement System One Decision Models

## Summary

This project develops a runnable tutorial, a small decision application, and
then a blog about typed probabilistic decisions. Phase 1 checks the Liquid
d1-3B decision API; Phase 2 teaches its typed questions in a paired notebook;
Phase 3 applies a fixed routing policy to synthetic tickets.

## Structure of the Dir

- `data/`: Fixed synthetic support tickets, labels, provenance, and license.
- `test/`: Focused checks for the frozen split and cost policy.

## Description of Files

- `dev_predictions.json`: Frozen development-split model probabilities and
  routing actions.
- `probe_decision_api.py`: Executable API contract and behavior probe.
- `probe_results.json`: Captured requests, responses, and observations from the
  pinned Phase 1 run.
- `README.md`: Setup, pinned versions, probe results, and API limits.
- `requirements.txt`: Pinned Python packages needed to fetch the model and
  run the notebook.
- `system-one.API.ipynb`: Guided API notebook with short live examples.
- `system-one.API.py`: Jupytext percent-format pair for the notebook.
- `system_one_API_utils.py`: Small HTTP client and probability displays.
- `test/test_ticket_routing.py`: Checks balanced disjoint splits and policy
  choices on known probability distributions.
- `ticket_routing.py`: Fixed ticket split, decision prompt, action costs, and
  command line prediction runner.

## Pinned Model and API

The primary model is [Liquid d1-3B](https://huggingface.co/LiquidAI/d1-3B),
served through llama.cpp's [`POST /v1/systemone`](https://github.com/ggml-org/llama.cpp/blob/8956c769e8ababd9fbd89c81764f241d5f515d68/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api)
decision endpoint. This is the model's typed decision interface: it accepts
one state and named `noul`, `choice`, or `score` questions, and returns
probabilities rather than generated prose.

| Component | Pinned version |
| :--- | :--- |
| Model repository | `LiquidAI/d1-3B-GGUF` |
| Model license | `LFM1.0` (see the model card before redistribution) |
| Model revision | `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563` |
| Model file | `d1-3B-Q8_0.gguf` |
| Model file SHA-256 | `2f0942d5a5f64cf69c3356d2be439b71644b1f0eb3a580912a5a0179eeaba77d` |
| llama.cpp commit | `8956c769e8ababd9fbd89c81764f241d5f515d68` |
| Python packages | See `requirements.txt` |
| Probe endpoint | `http://127.0.0.1:18089/v1/systemone` |

Q8 is a quantized copy of d1-3B. Its probabilities may differ from the
full-precision checkpoint, so results from this project should identify the
GGUF file and should not be treated as a replication of the model card's
calibration figures. The probe uses text only; images are outside Phase 1.

## Description of Executables

### `probe_decision_api.py`

The script sends a text state and a JSON state, each with three named
questions. It also sends a previously unseen choice schema, reverses option
order, repeats a request, and compares a choice asked alone with the same
choice asked alongside other questions. It validates answer types, choice
names, option probability keys and sums, score legends, and expected score.
The optional JSON report contains every request and response.

### `system-one.API.ipynb` and `system-one.API.py`

The notebook walks through `noul`, `choice`, and `score`; a JSON state; several
questions over one state; a new choice schema; and a top-probability action.
The Python pair can be executed from the repository root while the server is
running:

```bash
> python research/Implement_System_One_Decision_Models/system-one.API.py
```

Open `system-one.API.ipynb` in Jupyter for the cell-by-cell lesson. To sync
edits to either paired file, run:

```bash
> jupytext --sync research/Implement_System_One_Decision_Models/system-one.API.ipynb
```

### `ticket_routing.py`

The application predicts a route for each case in the fixed `dev` or `test`
split. It records the model's distribution, a top-probability route, and the
action selected by an expected-cost policy. The `review` action has a fixed
cost and sends a ticket to a human. The model estimates outcomes; the policy
turns those estimates into actions.

The 72 original synthetic cases are documented in [data/README.md](data/README.md).
Each split contains 12 cases per route. The prompt, label space, and costs
live as constants in `ticket_routing.py`; they were fixed before running the
test split. The illustrative cost units are not measured business costs:

| Action | True billing | True shipping | True technical |
| :--- | ---: | ---: | ---: |
| Route to billing | 0 | 5 | 7 |
| Route to shipping | 8 | 0 | 6 |
| Route to technical | 8 | 6 | 0 |
| Human review | 1.5 | 1.5 | 1.5 |

From the repository root, run one split while the server is listening:

```bash
> python research/Implement_System_One_Decision_Models/ticket_routing.py \
    --split dev --output /tmp/dev_predictions.json
> PYTHONPATH=helpers_root:. python -m pytest \
    research/Implement_System_One_Decision_Models/test/test_ticket_routing.py
```

The expected-cost action minimizes the row's probability-weighted cost. The
top-probability route is recorded separately for comparison. Do not change
the prompt or costs after opening the test split if you want to compare the
saved evaluation results.

On the 36 development cases, the top-probability route matched 34 labels.
The cost policy reviewed four cases and routed 31 of the remaining 32
correctly. These observations are saved in [dev_predictions.json](dev_predictions.json).
They are a prompt and integration check on synthetic data, not a held-out
performance claim. The prompt and cost table above are now frozen for the
test evaluation.

### Setup and Run

Use Python 3.12, a C/C++ compiler, and Git. Clone and build the pinned server
outside this repository. These commands use `/tmp/d1_phase1` as an example
working directory:

```bash
> python -m venv /tmp/d1_phase1/venv
> /tmp/d1_phase1/venv/bin/pip install -r requirements.txt
> git clone https://github.com/ggml-org/llama.cpp.git /tmp/d1_phase1/llama.cpp
> git -C /tmp/d1_phase1/llama.cpp checkout 8956c769e8ababd9fbd89c81764f241d5f515d68
> /tmp/d1_phase1/venv/bin/cmake -S /tmp/d1_phase1/llama.cpp \
    -B /tmp/d1_phase1/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF
> /tmp/d1_phase1/venv/bin/cmake --build /tmp/d1_phase1/llama.cpp/build \
    --target llama-server -j 4
```

Fetch the pinned model file without adding it to Git:

```bash
> /tmp/d1_phase1/venv/bin/python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id="LiquidAI/d1-3B-GGUF", filename="d1-3B-Q8_0.gguf", revision="bb1e436ea78eb96a3f1acb6da865f70c2fbeb563", local_dir="/tmp/d1_phase1/model"))'
> sha256sum /tmp/d1_phase1/model/d1-3B-Q8_0.gguf
```

On macOS, use `shasum -a 256` for the checksum command.

Start the server in one terminal, then run the probe in another:

```bash
> /tmp/d1_phase1/llama.cpp/build/bin/llama-server \
    -m /tmp/d1_phase1/model/d1-3B-Q8_0.gguf \
    --host 127.0.0.1 --port 18089 -c 2048
> /tmp/d1_phase1/venv/bin/python probe_decision_api.py \
    --output /tmp/d1_phase1/report.json
```

The script prints `"validated": true` when all six requests meet the API
contract. `observations` reports the largest option probability change after
reordering, whether a repeated call returned identical answers, and whether
asking one question alone changed its answer. These are integration checks,
not accuracy or calibration measurements.

### Phase 1 Result

The pinned Q8 model and server completed all six cases on 2026-10-30. The
text and JSON states each returned valid `noul`, `choice`, and `score`
answers; the new four-option schema also returned a valid distribution. The
full request and response record is in [probe_results.json](probe_results.json).

| Check | Observation |
| :--- | :--- |
| Repeated identical request | Identical answers |
| One choice alone versus in a three-question request | Identical answer and probabilities |
| Reversed option order in the same three-question request | Maximum probability change: 0.00510; top choice remained `billing` |

Each case was run once, so the response times in the JSON file are diagnostic
only. This small example tests the API contract and some representation
sensitivity; it does not measure general accuracy, calibration, or latency.

### Documented API Limits

The [d1-3B model card](https://huggingface.co/LiquidAI/d1-3B) lists a 32,768
token context and 2–10 levels for a `score` question. The pinned llama.cpp
server documents a maximum of 255 options for d1 `choice` questions. A
`state` may be a string or JSON value. The native Python API additionally
offers `system_one_batch` for many states; this HTTP endpoint accepts one
state per request and multiple independent named questions within that
request. Phase 1 checks the latter behavior, not multi-state batch throughput.
The example server command sets a 2,048 token context for these short inputs;
larger inputs require a correspondingly larger `-c` value.

The server scales probabilities using temperature values stored in the model
file. Valid probability ranges and sums do not establish calibration on a
specific dataset; that requires labeled cases in a later phase.
