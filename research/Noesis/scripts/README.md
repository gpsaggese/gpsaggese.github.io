# Scripts

- One-off scripts that measure the MVP's assumptions; they are not imported by any
  component

## Description of Files

| File                      | Description                                       |
| :------------------------ | :------------------------------------------------ |
| `detection_math.py`       | Exact false-alarm and detection probabilities     |
| `demo_contract_monitoring.py` | Exercise probe-to-verdict monitoring with deterministic in-memory boundaries (`PR_S4`) |
| `generate_canary_bank.py` | Reproducibly generate versioned canary candidates |
| `run_checker_replay.py` | Run production-checker replay, analytic comparison, and artifact generation |
| `spike_openrouter.py`     | Measure pinned OpenRouter providers for one model |
| `demo_supply_asks.py`     | Show OpenRouter prices becoming one ask per seller and clearing against a bid (`PR_S9`) |
| `demo_credit_ledger.py`   | Show the free credit ledger gating bids and charging cleared contracts (`PR_P4`) |

## Description of Executables

- Print the detection table:

  ```bash
  > scripts/detection_math.py
  ```

- Contract-monitoring demo (no key, database, network, or spend):

  ```bash
  > research/Noesis/scripts/demo_contract_monitoring.py
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
- Full checker replay demo (no key, database, network, or spend):
  ```bash
  > research/Noesis/scripts/run_checker_replay.py
  ```

## Auction Report Demo (PR_M12)

- From the repository root, display a reproducible auction (mocked fulfillment):
  ```bash
  > PYTHONPATH="$PWD/helpers_root:$PWD" python -m research.Noesis.scripts.demo_auction_report --scenario excess_demand
  ```
- Export a supply/demand plot without opening a window:
  ```bash
  > MPLBACKEND=Agg PYTHONPATH="$PWD/helpers_root:$PWD" python -m research.Noesis.scripts.demo_auction_report --scenario multi_tier --plot /tmp/noesis-auction.png
  ```
- Choose `balanced`, `excess_demand`, `excess_supply`, `multi_tier`, or `no_cross`;
  `--seed` changes values reproducibly. Multi-tier exports its first cleared tier.
- Displays inputs, mocked contracts, per-tier volumes, and residual orders.
  No credentials, services, database, payments, or provider calls are required.
- Run the new tests headlessly:
  ```bash
  > MPLBACKEND=Agg PYTHONPATH="$PWD/helpers_root:$PWD" python -m pytest research/Noesis/market/test/test_auction_report.py research/Noesis/market/test/test_auction_plot.py research/Noesis/scripts/test/test_demo_auction_report.py
  ```
