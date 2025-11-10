# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Create environment
conda create -n horovod_env python=3.9
conda activate horovod_env

# 2. Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install dependencies
pip install transformers datasets pyyaml tensorboard nltk rouge-score

# 4. Install Horovod (requires MPI and NCCL)
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]

# 5. Verify installation
python -c "import torch, horovod.torch, transformers; print('All packages installed')"
```

## Running Training

### Local (2+ GPUs)

```bash
# Edit NUM_GPUS in scripts/train_local.sh
bash scripts/train_local.sh
```

### Zaratan Cluster (msml610)

```bash
# 1. Submit job (uses H100 GPUs by default)
sbatch scripts/train_zaratan.sh

# Or with specific config:
sbatch scripts/train_zaratan.sh configs/h100.yaml

# 2. Monitor
squeue -u vikranth
tail -f logs/horovod_transformer-*.out

# 3. Cancel if needed
scancel <JOB_ID>
```

**Note**: Script auto-configures for Zaratan paths:
- Cache: `/scratch/zt1/project/msml610/vikranth/hf_cache`
- Checkpoints: `/scratch/zt1/project/msml610/vikranth/checkpoints`

## Configuration Files

- `configs/base.yaml` - Small model for testing
- `configs/custom_transformer.yaml` - Full custom model
- `configs/distilgpt2.yaml` - Fine-tune DistilGPT-2
- `configs/gpt2_small.yaml` - Fine-tune GPT-2
- `configs/h100.yaml` - Optimized for Zaratan H100 GPUs

## Monitoring

```bash
# TensorBoard
tensorboard --logdir runs/

# Training logs
tail -f logs/train_*.log
```

## Text Generation

```bash
# After training
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --model_type custom \
  --interactive

# Or use pretrained models
python -m src.generate \
  --model_type distilgpt2 \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```

## Notebooks

```bash
jupyter notebook notebooks/
```

1. `00_env_and_dataset.ipynb` - Environment and data exploration
2. `01_model_smoke_test.ipynb` - Model testing and demo

## Important Notes

1. **Requires 2+ GPUs** for distributed training
   - With 1 GPU: runs smoke test only
   - With 2+ GPUs: runs full training

2. **Horovod Installation**
   - Needs MPI (OpenMPI recommended)
   - Needs NCCL for GPU operations
   - See README.md for detailed instructions

3. **Dataset Download**
   - First run downloads BookCorpus (~7GB)
   - Set `HF_HOME` to control cache location
   - Falls back to synthetic data if download fails

4. **Zaratan Cluster**
   - Login via UMD VPN + SSH
   - Load appropriate modules
   - Set scratch directory for cache

## Common Issues

**"horovodrun not found"**
```bash
pip install horovod[pytorch]
```

**"Only 1 GPU detected"**
```bash
# Use horovodrun with correct GPU count
horovodrun -np 4 python -m src.train --config configs/base.yaml
```

**"CUDA out of memory"**
```yaml
# Reduce batch size in config
training:
  per_gpu_batch_size: 4  # down from 8
```

## Expected Results

- **Custom Model** (~30M params): Perplexity ~15-25 after 3 epochs
- **DistilGPT-2** (82M params): Perplexity ~10-15 after fine-tuning
- **GPT-2 Small** (117M params): Perplexity ~8-12 after fine-tuning

Training time: 2-4 hours on 4x A100 GPUs (full dataset)

## Verify Everything Works

```bash
# Test imports
python -c "from src.models.transformer_lm import TransformerLM; print('Models OK')"
python -c "from src.data import get_dataloaders; print('Data OK')"
python -c "from src.utils import setup_horovod; print('Utils OK')"

# Smoke test (1 GPU)
python -m src.train --config configs/base.yaml

# Should print: "Only 1 GPU detected — running smoke test"
```

## Next Steps

1. Run notebooks to explore data/models
2. Test local training with 2 GPUs
3. Submit Zaratan job for full training
4. Monitor training with TensorBoard
5. Generate text with trained model
6. Evaluate with metrics

For detailed information, see **README.md**.

