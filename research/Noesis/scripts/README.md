# Scripts

- One-off scripts that measure the MVP's assumptions; they are not imported by any
  component

## Description of Files

| File                  | Description                                                      |
| :-------------------- | :--------------------------------------------------------------- |
| `detection_math.py`   | Exact false-alarm and detection probabilities of the violation rules |
| `spike_openrouter.py` | Measure pinned OpenRouter providers for one model; writes `docs/findings/` |

## Description of Executables

- Print the detection table:

  ```bash
  > scripts/detection_math.py
  ```

- Dry run of the OpenRouter spike (no key, no spend):

  ```bash
  > research/Noesis/scripts/spike_openrouter.py --dry-run
  ```

- Real run, 20 calls per provider (needs `OPENROUTER_API_KEY`):

  ```bash
  > research/Noesis/scripts/spike_openrouter.py --n 20
  ```
