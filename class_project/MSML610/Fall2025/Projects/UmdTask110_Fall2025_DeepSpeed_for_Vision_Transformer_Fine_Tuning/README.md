# Distributed Vision Transformer Training

This project contains scripts for comparing PyTorch DistributedDataParallel (DDP) and DeepSpeed ZeRO Stage 3 for fine-tuning large Vision Transformers on ImageNet-1k.

## Project Structure

```
.
├── test_environment.py              # Test script for environment verification
├── train_standard_ddp.py            # Standard PyTorch DDP training
├── train_deepspeed_stage3.py        # DeepSpeed ZeRO Stage 3 training
├── deepspeed_config_stage3.json     # DeepSpeed configuration
├── submit.slurm                     # Slurm submission script template
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Prerequisites

1. **Hardware**: Multi-GPU system (tested with 4x GPUs)
2. **Software**: 
   - PyTorch with CUDA support
   - DeepSpeed
   - Transformers (Hugging Face)
   - WandB
3. **Dataset**: ImageNet-1k dataset organized as:
   ```
   imagenet/
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ...
   ```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Login to WandB** (run once):
   ```bash
   wandb login
   ```

3. **Download ImageNet dataset** and organize it in the expected directory structure.

## Quick Start

### 1. Test Environment

Verify your environment is set up correctly:

```bash
# Test without distributed training
python test_environment.py

# Test with distributed training (using torchrun)
torchrun --nproc_per_node=4 test_environment.py
```

### 2. Train with Standard PyTorch DDP

```bash
# Using torchrun
torchrun --nproc_per_node=4 train_standard_ddp.py \
    --data-path /path/to/imagenet \
    --batch-size 64 \
    --epochs 2 \
    --learning-rate 1e-4 \
    --num-workers 8

# Or using srun (if available)
srun python train_standard_ddp.py \
    --data-path /path/to/imagenet \
    --batch-size 64 \
    --epochs 2 \
    --learning-rate 1e-4 \
    --num-workers 8
```

### 3. Train with DeepSpeed

```bash
# Using deepspeed launcher
deepspeed --num_gpus=4 train_deepspeed_stage3.py \
    --data-path /path/to/imagenet \
    --deepspeed-config deepspeed_config_stage3.json \
    --epochs 2 \
    --num-workers 8

# Or using srun
srun deepspeed --num_gpus=4 train_deepspeed_stage3.py \
    --data-path /path/to/imagenet \
    --deepspeed-config deepspeed_config_stage3.json \
    --epochs 2 \
    --num-workers 8
```

### 4. Submit via Slurm

Edit `submit.slurm` to:
1. Update module loading commands for your cluster
2. Uncomment the training command you want to run
3. Update the data path

Then submit:
```bash
sbatch submit.slurm
```

Monitor the job:
```bash
tail -f logs/slurm_<JOB_ID>.out
```

## Key Features

### Standard PyTorch DDP (`train_standard_ddp.py`)

- Uses `torch.nn.parallel.DistributedDataParallel`
- BF16 mixed precision with `torch.cuda.amp.autocast`
- WandB logging for metrics and GPU utilization
- Supports command-line arguments for flexibility

### DeepSpeed Training (`train_deepspeed_stage3.py`)

- Uses DeepSpeed ZeRO Stage 3 for memory optimization
- BF16 mixed precision enabled in DeepSpeed config
- CPU offloading disabled (GPU-only)
- Gradient accumulation supported
- AdamW optimizer with learning rate scheduling

### DeepSpeed Configuration (`deepspeed_config_stage3.json`)

Key settings:
- **ZeRO Stage 3**: Partitions optimizer states, gradients, and parameters
- **BF16 enabled**: For better performance on modern GPUs
- **Micro batch size**: 8 per GPU
- **Gradient accumulation**: 4 steps
- **No offloading**: Keeps everything on GPU for better speed

## Performance Comparison

The scripts log the following metrics to WandB:

- Training loss
- Validation accuracy  
- Epoch duration
- GPU memory usage (allocated and reserved)
- GPU utilization

Compare these metrics between the two approaches to evaluate:
- Training speed (epochs/second)
- Memory efficiency (GPU usage)
- Final model accuracy

## Command Line Arguments

### Standard DDP Training

```bash
python train_standard_ddp.py \
    --data-path PATH          # Path to ImageNet dataset (required)
    --batch-size INT          # Batch size per GPU (default: 64)
    --epochs INT              # Number of epochs (default: 1)
    --learning-rate FLOAT     # Learning rate (default: 1e-4)
    --weight-decay FLOAT      # Weight decay (default: 0.01)
    --num-workers INT         # Data loader workers (default: 4)
```

### DeepSpeed Training

```bash
python train_deepspeed_stage3.py \
    --data-path PATH          # Path to ImageNet dataset (required)
    --deepspeed-config PATH   # DeepSpeed config file (default: deepspeed_config_stage3.json)
    --epochs INT              # Number of epochs (default: 1)
    --num-workers INT         # Data loader workers (default: 4)
    --local_rank INT          # Local rank (auto-detected, default: -1)
```

## Troubleshooting

### CUDA Out of Memory

1. **Reduce batch size**: Decrease `--batch-size` for DDP or `train_micro_batch_size_per_gpu` in DeepSpeed config
2. **Enable gradient accumulation**: DeepSpeed config already includes this
3. **Use gradient checkpointing**: Add to model configuration

### NCCL Errors

1. Check that NCCL is properly installed
2. Try setting `export NCCL_DEBUG=INFO` for more details
3. Ensure proper Slurm/SGE configuration for multi-GPU jobs

### WandB Connection Issues

1. Run `wandb login` if not already done
2. Set `WANDB_MODE=disabled` to run without WandB if needed
3. Check network connectivity to WandB servers

## Model Details

- **Model**: `google/vit-large-patch16-224`
- **Parameters**: ~307M
- **Input size**: 224x224
- **Num classes**: 1000 (ImageNet-1k)

## References

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Transformers Library](https://huggingface.co/docs/transformers/index)

## License

See LICENSE file in the repository.
Testing.

