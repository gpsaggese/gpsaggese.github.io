# Training a Custom Transformer Language Model from Scratch

**UMD MSML610 Course Project** — Train a transformer-based language model from scratch on BookCorpus dataset using Horovod for distributed multi-GPU training on the Zaratan HPC cluster.

---

## 🎯 Project Goal

Train a **custom transformer language model from scratch** on the BookCorpus dataset. The model is a decoder-only (GPT-like) transformer architecture implemented from scratch.

---

## 📋 Quick Start

### 1. **Preprocess Data** (Run once)
```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```
This notebook downloads BookCorpus, tokenizes it, and saves preprocessed data to `data/preprocessed/`.

### 2. **Train Model** (On Zaratan HPC)
```bash
sbatch scripts/train_zaratan.sh
```

### 3. **Generate Text**
```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/config.yaml \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```

---

## 📁 Project Structure

```
.
├── notebooks/
│   └── 00_data_preprocessing.ipynb    # Data preprocessing (download, tokenize, split)
│
├── src/
│   ├── models/
│   │   └── transformer_lm.py          # Custom transformer model (GPT-like)
│   ├── data.py                         # Load preprocessed data
│   ├── train.py                        # Distributed training script
│   ├── generate.py                     # Text generation
│   ├── metrics.py                      # Evaluation metrics
│   └── utils/                          # Utilities (config, logging, distributed)
│
├── configs/
│   └── config.yaml                     # Single configuration file
│
├── scripts/
│   ├── train_local.sh                  # Local multi-GPU training
│   └── train_zaratan.sh                # Slurm job script for Zaratan
│
├── data/
│   └── preprocessed/                   # Preprocessed data (created by notebook)
│       ├── train/                      # Training data
│       ├── val/                        # Validation data
│       └── tokenizer/                  # Saved tokenizer
│
├── checkpoints/                        # Model checkpoints (created during training)
├── logs/                               # Training logs
└── runs/                               # TensorBoard logs
```

---

## 🔧 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- OpenMPI (for Horovod)
- NCCL (for multi-GPU training)

### Install Dependencies
```bash
pip install -r requirements.txt

# Install Horovod with GPU support
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
```

---

## 📊 Workflow

### Step 1: Data Preprocessing

Run the Jupyter notebook to prepare data:
```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```

This will:
1. Download BookCorpus dataset (~7GB)
2. Tokenize with GPT-2 tokenizer
3. Create train/validation splits (95%/5%)
4. Pack tokens into fixed-length blocks (reduces padding)
5. Save to `data/preprocessed/`

**Note**: Run this once before training. The preprocessed data will be reused for all training runs.

### Step 2: Training

#### Local Training (2+ GPUs)
```bash
bash scripts/train_local.sh
```

#### Zaratan HPC Cluster
```bash
sbatch scripts/train_zaratan.sh
```

The training script will:
- Load preprocessed data from `data/preprocessed/`
- Create custom transformer model
- Train with Horovod distributed training
- Save checkpoints to `checkpoints/`
- Log metrics to TensorBoard (`runs/`)

### Step 3: Text Generation

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/config.yaml \
  --prompt "In a galaxy far, far away" \
  --max_new_tokens 150 \
  --temperature 0.8 \
  --interactive
```

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Model**: Architecture (d_model, n_layers, n_heads, etc.)
- **Training**: Epochs, batch size, learning rate, mixed precision
- **Data**: Path to preprocessed data directory

---

## 🏗️ Model Architecture

**Custom Transformer Language Model** (Decoder-only, GPT-like):

- **Causal self-attention** (autoregressive)
- **Sinusoidal positional encoding**
- **Multi-head attention** (12 heads)
- **Feed-forward networks** with GELU activation
- **Layer normalization** (pre-norm)
- **Weight tying** (embedding & output projection)

**Default Configuration**:
- Parameters: ~150M
- d_model: 768
- Layers: 12
- Heads: 12
- d_ff: 3072
- Max sequence length: 512

---

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Training Logs
```bash
tail -f logs/train_*.log
```

### Structured Metrics
Training metrics are saved to `runs/structured/<timestamp>_run/`:
- `metrics.csv` - Training/validation metrics
- `run_metadata.json` - Run metadata
- `config_used.yaml` - Configuration used

Generate HTML report:
```bash
python scripts/generate_report.py --run_dir runs/structured/<timestamp>_run
```

---

## 🖥️ Zaratan HPC Setup

### Cluster Configuration
- **Account**: `msml610-class`
- **Partition**: `gpu` (H100 GPUs)
- **GPUs**: 4x H100 (80GB VRAM each)
- **User**: `vikranth`

### Environment Setup
The training script (`train_zaratan.sh`) automatically:
- Loads required modules (CUDA, Python, OpenMPI, NCCL)
- Activates virtual environment
- Sets up environment variables
- Configures Horovod for single-node multi-GPU training

### Submit Job
```bash
sbatch scripts/train_zaratan.sh
```

### Monitor Job
```bash
# Check status
squeue -u vikranth

# View output
tail -f logs/horovod_transformer_h100-<JOB_ID>.out
```

---

## 🐛 Troubleshooting

### "Preprocessed data not found"
**Solution**: Run `notebooks/00_data_preprocessing.ipynb` first.

### "Only 1 GPU detected"
**Solution**: This project requires 2+ GPUs for distributed training. Allocate more GPUs:
```bash
horovodrun -np 4 -H localhost:4 python -m src.train --config configs/config.yaml
```

### "CUDA out of memory"
**Solution**: Reduce batch size in `configs/config.yaml`:
```yaml
training:
  per_gpu_batch_size: 8  # Reduce from 16
```

### "Horovod not found"
**Solution**: Install Horovod:
```bash
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
```

---

## 📚 Key Features

- ✅ **Single custom model** - Train from scratch, no pretrained models
- ✅ **Data preprocessing in notebook** - All data preparation in one place
- ✅ **Distributed training** - Horovod for multi-GPU training
- ✅ **H100 optimized** - BF16 mixed precision, large batches
- ✅ **Comprehensive logging** - TensorBoard + structured CSV metrics
- ✅ **Text generation** - Interactive and batch generation

---

## 📝 License

This project is for educational purposes as part of UMD MSML610 coursework.

---

## 🙏 Acknowledgments

- UMD Division of IT for Zaratan HPC cluster access
- HuggingFace for datasets and tokenizers
- Horovod team for distributed training framework
