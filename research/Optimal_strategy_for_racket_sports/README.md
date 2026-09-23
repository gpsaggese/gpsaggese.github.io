# Optimal Shot Placement Framework for Racket Sports

Implementation of the reduced-order framework for shot placement in tennis and pickleball from the paper "Optimal Strategy for Racket Sports."

## Quick Start

### Build Docker Image

```bash
./docker_build.sh
```

### Run Jupyter Lab

```bash
./docker_jupyter.sh
```

### Run Tests

```bash
./docker_exec.sh pytest test/
```

## Module Structure

Modules are organized in layers, with each layer importing only from layers above it:

| Module | Purpose |
|--------|---------|
| `racket_params.py` | Sport, player, and error parameters; court geometry |
| `racket_trajectory.py` | Closed-form 1D/2D ball flight and error propagation |
| `racket_scoring.py` | Grid, Monte Carlo in-bounds probability, reachability, score |
| `racket_game.py` | Zero-sum shot placement game |
| `racket_strategy_utils.py` | Plotting and experiment helpers for notebooks |

## Development

Requirements are in `requirements.txt`. Add scipy for optimization and numerical integration.

```bash
pip install -r requirements.txt
```

Run tests locally:

```bash
pytest test/
```

## References

- Paper: "Optimal Strategy for Racket Sports" (Sections III-VII)
- Data: Table I (sport parameters), Figure 1 (trajectory validation)
