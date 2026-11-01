# Implement System One Decision Models

## Summary

This project develops a runnable tutorial, a small decision application, and
then a blog about typed probabilistic decisions. Phase 1 checks the Liquid
d1-3B decision API with a small executable probe.

## Structure of the Dir

This phase has no subdirectories. Later phases will add the paired notebook,
application, and evaluation files.

## Description of Files

- `README.md`: Setup, pinned versions, probe results, and API limits.
- `probe_decision_api.py`: Executable API contract and behavior probe.
- `probe_results.json`: Captured requests, responses, and observations from the
  pinned Phase 1 run.
- `requirements.txt`: Pinned Python packages needed to fetch the model and
  build the server.

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
