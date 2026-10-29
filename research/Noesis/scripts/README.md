# Scripts

- One-off scripts that measure the MVP's assumptions; they are not imported by any
  component

## Description of Files

| File                      | Description                                       |
| :------------------------ | :------------------------------------------------ |
| `detection_math.py`       | Exact false-alarm and detection probabilities     |
| `generate_canary_bank.py` | Reproducibly generate versioned canary candidates |
| `spike_openrouter.py`     | Measure pinned OpenRouter providers for one model |
| `demo_supply_asks.py`     | Show OpenRouter prices becoming one ask per seller and clearing against a bid (`PR_S9`) |
| `demo_credit_ledger.py`   | Show the free credit ledger gating bids and charging cleared contracts (`PR_P4`) |

## Description of Executables

- Print the detection table:

  ```bash
  > scripts/detection_math.py
  ```

- Regenerate the v1 canary candidate bank:

  ```bash
  > python3 research/Noesis/scripts/generate_canary_bank.py \
      --output-dir research/Noesis/data/canaries
  ```

- Dry run of the OpenRouter spike (no key, no spend):

  ```bash
  > research/Noesis/scripts/spike_openrouter.py --dry-run
  ```

- Real run, 20 calls per provider (needs `OPENROUTER_API_KEY`):

  ```bash
  > research/Noesis/scripts/spike_openrouter.py --n 20
  ```
- Supply adapter demo on the checked-in catalog sample (no key, no network):
  ```bash
  > research/Noesis/scripts/demo_supply_asks.py
  ```
- Same against the live OpenRouter catalog (read-only, no spend; reads
  `OPENROUTER_API_KEY` from the env or `research/Noesis/.env`):
  ```bash
  > research/Noesis/scripts/demo_supply_asks.py --live
  ```
- Credit ledger demo over an in-process app (no key, no network):
  ```bash
  > research/Noesis/scripts/demo_credit_ledger.py
  ```
