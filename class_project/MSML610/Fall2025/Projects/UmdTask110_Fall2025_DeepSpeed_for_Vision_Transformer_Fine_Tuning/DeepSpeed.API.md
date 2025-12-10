# DeepSpeed API Documentation

Minimal documentation of the DeepSpeed API and wrapper functions.

## Core DeepSpeed Concepts

DeepSpeed provides memory-efficient distributed training through:
- **ZeRO (Zero Redundancy Optimizer)**: Partitions optimizer states, gradients, and parameters across GPUs
- **Mixed Precision**: BF16/FP16 support
- **CPU Offloading**: Offload optimizer states and parameters to CPU

## Key DeepSpeed APIs

### Initialization

```python
import deepspeed

model_engine, optimizer, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config_path
)
```

### Model Engine

The DeepSpeed model engine extends PyTorch models:

```python
# Forward pass
outputs = model_engine(inputs)

# Backward pass
model_engine.backward(loss)

# Optimizer step
model_engine.step()
```

## Wrapper Functions

All reusable functions are in `DeepSpeed_FSDP_utils.py`:

- `load_vit_model()`: Load Vision Transformer models
- `create_deepspeed_config()`: Create DeepSpeed configuration
- `initialize_deepspeed_model()`: Initialize DeepSpeed engine
- `train_epoch_deepspeed()`: Training loop for DeepSpeed
- `evaluate_deepspeed()`: Evaluation with DeepSpeed
- `run_training_experiment()`: Complete experiment runner (handles all methods)

See `DeepSpeed_FSDP.API.ipynb` for minimal interactive examples.

## ZeRO Stages

- **ZeRO Stage 0**: Baseline DDP (no optimization)
- **ZeRO Stage 1**: Partitions optimizer states
- **ZeRO Stage 2**: Partitions optimizer states and gradients
- **ZeRO Stage 3**: Partitions optimizer states, gradients, and parameters

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
