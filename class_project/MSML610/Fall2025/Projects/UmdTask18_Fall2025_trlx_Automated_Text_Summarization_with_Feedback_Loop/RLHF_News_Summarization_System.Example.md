# RLHF News Summarization System - Example Notebook

This document provides comprehensive documentation for the RLHF News Summarization System example notebook.

## Overview

The example notebook (`RLHF_News_Summarization_System.example.ipynb`) demonstrates the complete pipeline for building an RLHF-optimized news summarization system, from data preparation through DPO training and evaluation.

## Purpose

This notebook serves as a comprehensive reference implementation that:
- Explains each stage of the RLHF pipeline in detail
- Demonstrates key concepts without requiring full retraining
- Provides code examples and explanations for reviewers
- Shows the complete workflow from raw data to RLHF-optimized model

## Pipeline Stages

### Stage 1: Data Preparation
- Load CNN/DailyMail dataset from Hugging Face
- Clean and preprocess articles and summaries
- Tokenize using T5 tokenizer
- Create train/validation/test splits
- Save processed dataset for reuse

### Stage 2: Supervised Fine-Tuning (SFT)
- Train T5-small baseline model
- Full fine-tuning on CNN/DailyMail
- Evaluate with ROUGE metrics
- Save baseline checkpoint

### Stage 3: LoRA Fine-Tuning
- Apply LoRA (Low-Rank Adaptation) to T5-large
- Parameter-efficient training
- Train BART-large with LoRA for comparison
- Compare T5-large vs BART-large performance
- Merge LoRA weights into base model

### Stage 4: Preference Pair Generation
- Load SFT models (T5-small and T5-large)
- Generate summaries for same articles
- Create preference pairs (chosen vs rejected)
- Save as JSONL for DPO training

### Stage 5: DPO Training
- Configure TRLX with custom DPO trainer
- Train on preference pairs
- Optimize for human-aligned summaries
- Save RLHF-optimized model

### Stage 6: Evaluation
- Compare baseline vs RLHF models
- Compute ROUGE scores
- Qualitative assessment
- Demonstrate improvements

## Key Concepts Explained

### What is RLHF?
Reinforcement Learning from Human Feedback (RLHF) is a technique to align language models with human preferences. Instead of just predicting the next token, the model learns to generate outputs that humans prefer.

### Why DPO Instead of PPO?
- PPO (Proximal Policy Optimization) requires decoder-only models
- T5 is an encoder-decoder architecture
- DPO (Direct Preference Optimization) works with seq2seq models
- DPO is simpler and more stable than PPO

### Preference Pair Generation
We use AI feedback (RLAIF) instead of human annotation:
- T5-large summaries = "chosen" (higher quality)
- T5-small summaries = "rejected" (lower quality)
- This simulates human preference without manual labeling

### LoRA Benefits
- Trains only a small number of parameters
- Reduces memory requirements
- Faster training than full fine-tuning
- Can be merged back into base model

## Prerequisites

Before running the notebook:

1. **Python Environment**:
   - Python 3.10+
   - All dependencies from `requirements.txt`
   - Virtual environment recommended

2. **Hardware**:
   - 16GB+ RAM recommended
   - GPU/MPS optional but recommended
   - CPU training is supported but slower

3. **Disk Space**:
   - 10GB+ for models and datasets
   - Additional space for checkpoints

## Running the Notebook

### Full Pipeline (Training)
If you want to train all models from scratch:
```bash
# This will take several hours
jupyter notebook RLHF_News_Summarization_System.example.ipynb
# Run all cells sequentially
```

### Demo Mode (Using Pre-trained Models)
If models are already trained:
- Skip training cells
- Load pre-trained checkpoints
- Run evaluation and demo cells only

## Notebook Structure

The notebook is organized into clear sections:

1. **Introduction**: Project overview and objectives
2. **Setup**: Imports and configuration
3. **Data Preparation**: CNN/DailyMail processing
4. **SFT Training**: T5-small baseline
5. **LoRA Training**: T5-large with LoRA
6. **Preference Generation**: Create DPO dataset
7. **DPO Training**: RLHF optimization
8. **Evaluation**: Compare all models
9. **Demo**: Interactive summarization
10. **Summary**: Key findings and next steps

## Expected Outputs

### Data Preparation
- Processed dataset in `data/processed/t5-small-512/`
- Train: 14,355 examples
- Validation: 668 examples
- Test: 574 examples

### SFT Training
- Model checkpoint in `data/models/t5-small-baseline/`
- ROUGE-1: ~0.41
- ROUGE-2: ~0.19
- ROUGE-L: ~0.29

### LoRA Training

T5-large with LoRA:
- Model checkpoint in `data/models/t5-large-merged/`
- ROUGE-1: ~0.45
- ROUGE-2: ~0.22
- ROUGE-L: ~0.32

BART-large with LoRA:
- Model checkpoint in `data/models/BART-large/`
- ROUGE-1: ~0.44
- ROUGE-2: ~0.21
- ROUGE-L: ~0.31

Winner: T5-large (better across all metrics)

### DPO Training
- Model checkpoint in `data/models/RLHF-t5-large-merged-dpo/`
- 400 preference pairs generated
- Improved alignment with human preferences

## Key Findings

1. **SFT Baseline**: T5-small achieves reasonable performance with full fine-tuning (ROUGE-L: 0.29)
2. **LoRA Efficiency**: T5-large with LoRA outperforms T5-small with fewer trainable parameters (ROUGE-L: 0.32, +10%)
3. **T5 vs BART**: T5-large outperforms BART-large across all ROUGE metrics (T5: 0.32 vs BART: 0.31 ROUGE-L)
4. **DPO Improvement**: RLHF training provides additional gains over LoRA baseline (ROUGE-L: 0.33, +3%)
5. **Model Ranking**: T5-large-DPO > T5-large-LoRA > BART-large-LoRA > T5-small-SFT

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Switch to CPU (slower but works)

### Slow Training
- Use GPU/MPS if available
- Reduce dataset size (use subset_frac parameter)
- Use LoRA instead of full fine-tuning

### Import Errors
- Ensure all dependencies installed
- Check Python version (3.10+)
- Activate virtual environment

## Integration with Other Notebooks

For detailed implementation of specific stages:
- `notebooks/data_preparation_and_baseline_t5.ipynb` - Data and SFT details
- `notebooks/lora_comparison.ipynb` - LoRA implementation and comparison
- `notebooks/RLHF_DPO.ipynb` - Complete DPO training workflow

## References

- CNN/DailyMail Dataset: [Hugging Face](https://huggingface.co/datasets/abisee/cnn_dailymail)
- T5 Model: [Google Research](https://arxiv.org/abs/1910.10683)
- TRLX Library: [CarperAI/trlx](https://github.com/CarperAI/trlx)
- DPO Paper: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- LoRA Paper: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
