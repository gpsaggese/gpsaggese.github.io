# Step-by-Step Training Guide

This guide provides clear instructions for starting distributed training on the Zaratan HPC cluster.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Access to Zaratan HPC cluster
- [ ] Account: `msml610` allocation
- [ ] SSH access to Zaratan login node
- [ ] Project files uploaded to cluster
- [ ] Python environment with dependencies installed

---

## Step 1: Connect to Zaratan HPC

```bash
# SSH into Zaratan login node
ssh vikranth@zaratan.umd.edu

# Navigate to your project directory
cd /afs/shell.umd.edu/project/msml610/user/vikranth/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation
# OR if using scratch space:
cd /scratch/zt1/project/msml610/vikranth/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation
```

---

## Step 2: Verify Environment Setup

### 2.1 Check Python Environment

```bash
# Check Python version (should be 3.9+)
python --version

# Check if required packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import horovod.torch as hvd; print(f'Horovod: {hvd.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2.2 Run Setup Verification (Optional)

```bash
# Run the verification script
python verify_setup.py
```

This will check:
- Python version
- PyTorch installation
- Horovod installation
- Project structure
- Module imports

---

## Step 3: Review and Configure Training Parameters

### 3.1 Check Available Configuration Files

```bash
ls -la configs/
```

Available configs:
- `base.yaml` - Default configuration (custom transformer, 3 epochs)
- `custom_transformer.yaml` - Custom model configuration
- `distilgpt2.yaml` - DistilGPT-2 fine-tuning
- `gpt2_small.yaml` - GPT-2 small fine-tuning
- `h100.yaml` - Optimized for H100 GPUs

### 3.2 Review Base Configuration

```bash
cat configs/base.yaml
```

Key settings in `base.yaml`:
- **Model**: Custom transformer (512 dim, 6 layers, 8 heads)
- **Training**: 3 epochs, batch size 8 per GPU, learning rate 5e-4
- **Dataset**: BookCorpus Open (full dataset)
- **Time**: Estimated 3-6 hours for 3 epochs

### 3.3 (Optional) Modify Configuration

If you want to change settings, edit the config file:

```bash
# Example: Use a smaller dataset for faster testing
nano configs/base.yaml
# Change: max_samples: null  to  max_samples: 10000
```

---

## Step 4: Prepare Directories

### 4.1 Create Required Directories

```bash
# These should already exist, but verify:
mkdir -p checkpoints
mkdir -p logs
mkdir -p runs
```

### 4.2 Check Disk Space

```bash
# Check available space in scratch (where datasets/cache will be stored)
df -h /scratch/zt1/project/msml610/vikranth/
```

Ensure you have at least 50GB free for:
- Dataset cache (~10-20GB)
- Model checkpoints (~1-5GB)
- Training logs (~100MB)

---

## Step 5: Review Training Script

### 5.1 Check Slurm Script Settings

```bash
# View the first 50 lines of the training script
head -50 scripts/train_zaratan.sh
```

Key settings:
- **GPUs**: 4x H100 (80GB)
- **Time**: 12 hours maximum
- **Memory**: 64GB per node
- **Partition**: gpu-h100 (or gpu-h100-hi)
- **Output**: `logs/horovod_transformer_h100-<JOB_ID>.out`

### 5.2 (Optional) Modify Script Settings

If needed, edit the script:

```bash
nano scripts/train_zaratan.sh
```

Common modifications:
- Change GPU type (line 25-26)
- Adjust time limit (line 29)
- Change memory allocation (line 28)
- Uncomment email notifications (lines 33-34)

---

## Step 6: Submit Training Job

### 6.1 Basic Submission (Using base.yaml)

```bash
# Submit job with default configuration
sbatch scripts/train_zaratan.sh configs/base.yaml
```

Expected output:
```
Submitted batch job 12345678
```

**Save the job ID** (e.g., `12345678`) - you'll need it for monitoring.

### 6.2 Alternative: Submit with Different Config

```bash
# Use H100 optimized config
sbatch scripts/train_zaratan.sh configs/h100.yaml

# Or use DistilGPT-2 config
sbatch scripts/train_zaratan.sh configs/distilgpt2.yaml
```

---

## Step 7: Monitor Job Status

### 7.1 Check Job Queue

```bash
# Check your job status
squeue -u vikranth

# Or check specific job
squeue -j <JOB_ID>
```

Status meanings:
- **PENDING (PD)**: Waiting for resources
- **RUNNING (R)**: Currently executing
- **COMPLETED (CD)**: Finished successfully
- **FAILED (F)**: Job failed

### 7.2 View Job Details

```bash
# Get detailed information about your job
scontrol show job <JOB_ID>
```

This shows:
- Node allocation
- GPU assignment
- Start/end times
- Resource usage

---

## Step 8: Monitor Training Progress

### 8.1 View Real-Time Output

```bash
# Watch the output file in real-time (replace <JOB_ID> with your job ID)
tail -f logs/horovod_transformer_h100-<JOB_ID>.out
```

Press `Ctrl+C` to stop watching.

### 8.2 Check Training Logs

```bash
# View the last 50 lines
tail -50 logs/horovod_transformer_h100-<JOB_ID>.out

# View entire log file
less logs/horovod_transformer_h100-<JOB_ID>.out
```

### 8.3 What to Look For

Good signs:
```
[INFO] Horovod initialized successfully:
  - World size: 4
  - Using CUDA: True
[INFO] Loading data...
[INFO] Dataset loaded and tokenized successfully.
[INFO] Created model...
[INFO] Starting distributed training:
Epoch 1 | Step 100/5000 | Loss: 8.2341 | PPL: 3782.45 | LR: 0.000500
```

Warning signs:
```
[ERROR] Horovod not installed
[ERROR] CUDA out of memory
[ERROR] Failed to load dataset
```

---

## Step 9: Check Training Results

### 9.0 Run Recording and HTML Report

In addition to TensorBoard, each training run records structured metrics and metadata for grading.

- Location (rank 0): `runs/structured/<timestamp>_run/`
- Contents: `metrics.csv`, `run_metadata.json`, `config_used.yaml`
- To generate a static HTML report with plots:

```bash
python scripts/generate_report.py --run_dir runs/structured/<timestamp>_run
```

Open `report.html` from that folder.

### 9.1 Verify Checkpoints

```bash
# List saved checkpoints
ls -lh checkpoints/

# Expected files:
# - best_model.pt (best validation loss)
# - checkpoint_epoch_0.pt
# - checkpoint_epoch_1.pt
# - checkpoint_epoch_2.pt
# - final_model.pt
```

### 9.2 Check Training Logs

```bash
# View training log files
ls -lh logs/

# View the training log
cat logs/train_*.log
```

### 9.3 View TensorBoard Logs

```bash
# Check if TensorBoard logs exist
ls -lh runs/

# To view TensorBoard (if you have port forwarding set up):
# tensorboard --logdir runs/ --port 6006
```

---

## Step 10: Handle Common Issues

### Issue 1: Job Pending Too Long

```bash
# Check queue status
squeue -u vikranth

# Check partition availability
sinfo -p gpu

# Consider reducing GPU count or time limit
```

### Issue 2: Out of Memory Error

```bash
# Reduce batch size in config
nano configs/base.yaml
# Change: per_gpu_batch_size: 8  to  per_gpu_batch_size: 4
```

### Issue 3: Dataset Download Fails

The script automatically falls back to synthetic data. To fix:
```bash
# Set HuggingFace cache directory
export HF_HOME=/scratch/zt1/project/msml610/vikranth/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export DATASET_CACHE=$HF_HOME
```

### Issue 4: Job Fails Immediately

```bash
# Check the error in output file
cat logs/horovod_transformer_h100-<JOB_ID>.out

# Common causes:
# - Horovod not installed
# - Wrong Python environment
# - Missing dependencies
```

---

## Step 11: Cancel or Modify Job

### 11.1 Cancel a Running Job

```bash
scancel <JOB_ID>
```

### 11.2 Check Job History

```bash
# View completed jobs
sacct -u vikranth --starttime=today

# View specific job details
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

---

## Step 12: After Training Completes

### 12.1 Verify Training Completed Successfully

```bash
# Check the end of output file
tail -100 logs/horovod_transformer_h100-<JOB_ID>.out

# Look for:
# "Training complete!"
# "Training finished with exit code: 0"
```

### 12.2 Check Final Results

```bash
# View final checkpoint
ls -lh checkpoints/final_model.pt

# Check training metrics in log
grep "Val Loss" logs/train_*.log
grep "Perplexity" logs/train_*.log
```

### 12.3 Generate Text (Optional)

```bash
# Test text generation with trained model
python -m src.generate \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --model_type custom \
    --prompt "Once upon a time" \
    --max_new_tokens 100
```

---

## Quick Reference Commands

```bash
# Submit job
sbatch scripts/train_zaratan.sh configs/base.yaml

# Check status
squeue -u vikranth

# Monitor output
tail -f logs/horovod_transformer_h100-<JOB_ID>.out

# Cancel job
scancel <JOB_ID>

# Check checkpoints
ls -lh checkpoints/

# View logs
cat logs/train_*.log
```

---

## Expected Training Timeline

For `configs/base.yaml` with 4 GPUs:

- **Dataset Loading**: 5-15 minutes (first time, cached after)
- **Epoch 1**: 1-2 hours
- **Epoch 2**: 1-2 hours
- **Epoch 3**: 1-2 hours
- **Total**: 3-6 hours

*Times vary based on dataset size, GPU type, and cluster load.*

---

## Next Steps After Training

1. **Evaluate Model**: Check validation loss and perplexity
2. **Generate Text**: Test generation quality
3. **Compare Models**: Try different configs (DistilGPT-2, GPT-2)
4. **Visualize Results**: Use TensorBoard to view training curves
5. **Save Results**: Copy important checkpoints/logs to permanent storage

---

## Getting Help

If you encounter issues:

1. Check the output log: `logs/horovod_transformer_h100-<JOB_ID>.out`
2. Review training logs: `logs/train_*.log`
3. Check `README.md` for troubleshooting section
4. Review `QUICKSTART.md` for common issues
5. Contact course TAs or cluster support

---

**Good luck with your training!**
