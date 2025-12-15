# Animal Face Classification with Transfer Learning and Ensemble Methods

A comprehensive deep learning project for classifying animal faces (cats, dogs, and wild animals) using transfer learning and ensemble methods. The project compares single model performance against homogeneous and heterogeneous ensemble approaches, with complete experiment tracking via Weights & Biases.

This project follows the **MSML610 Fall2025 class project template** and uses a modular utility script for all data preparation, model training, and evaluation tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Setup with Docker](#setup-with-docker)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results & Logging](#results--logging)
- [Configuration](#configuration)

## Overview

This project implements a complete machine learning pipeline for animal face classification using state-of-the-art transfer learning techniques. It explores three different modeling approaches:

1. **Single Model**: Individual pre-trained CNN architectures
2. **Homogeneous Ensemble**: Multiple instances of the same architecture (ResNet50)
3. **Heterogeneous Ensemble**: Combination of different architectures (ResNet50, EfficientNetB0, MobileNetV2)

The project includes comprehensive data augmentation, experiment tracking, and Docker containerization for reproducible training environments.

## Features

- **Transfer Learning**: Leverages pre-trained models (ResNet50, EfficientNetB0, MobileNetV2)
- **Data Augmentation**: Advanced augmentation techniques for improved generalization
- **Ensemble Methods**: Both homogeneous and heterogeneous ensemble implementations
- **Experiment Tracking**: Complete integration with Weights & Biases for monitoring
- **Docker Support**: Containerized environment for Jupyter notebooks
- **GPU Acceleration**: Automatic GPU detection and utilization (optimized for Google Colab T4 GPU)
- **Reproducible Results**: Consistent training pipeline with detailed logging

## Dataset

**AFHQ (Animal Faces-HQ)** dataset from Kaggle:
- **Source**: `andrewmvd/animal-faces`
- **Classes**: 3 (cat, dog, wild)
- **Split**: Pre-split into training and validation sets
- **Image Size**: 128x128 pixels
- **Format**: RGB images
- **Preprocessing**: Handled via `utils.py`

### Dataset Structure
```
afhq/
├── train/
│   ├── cat/
│   ├── dog/
│   └── wild/
└── val/
    ├── cat/
    ├── dog/
    └── wild/
```

### Data Augmentation
- Rescaling (normalization): 1/255
- Shear transformations: 0.2 range
- Zoom: 0.2 range
- Horizontal flipping
- Batch size: 256

## Architecture

### Supported Models

1. **ResNet50**
   - Pre-trained on ImageNet
   - Fine-tuned last 50 layers
   - Used for single model and homogeneous ensemble

2. **EfficientNetB0**
   - Efficient architecture with compound scaling
   - Balanced accuracy and efficiency

3. **MobileNetV2**
   - Lightweight architecture for faster inference
   - Depthwise separable convolutions

### Model Configuration
- **Input Shape**: (128, 128, 3)
- **Trainable Layers**: Last 50 layers (configurable)
- **Optimizer**: Adam
- **Learning Rate**: 0.00001 (default)
- **Loss Function**: Categorical Cross-Entropy
- **Epochs**: 20 (default)

### Ensemble Strategy
- **Voting Method**: Soft voting (average predictions)
- **Homogeneous**: 3 × ResNet50 models
- **Heterogeneous**: ResNet50 + EfficientNetB0 + MobileNetV2

## Setup with Docker

All dependencies are installed inside the Docker image. No local `pip install` is required.

```bash
# Clone the repository
git clone https://github.com/gpsaggese-org/umd_classes.git
cd "class_project/MSML610/Fall2025/Projects/UmdTask50_Fall2025_Weights_&_Biases_Image_Classification_with_Transfer_Learning"

# Build the Docker image
./docker_build.sh

# Launch Jupyter Notebook inside the container
./docker_jupyter.sh
```

### Docker Notes
- The container mounts your project folder, so all notebooks and scripts are accessible
- W&B metrics will work if `.env` with your `WANDB_API_KEY` is present in the project folder
- You can map a different host port for Jupyter by passing `-p <port>` to `docker_jupyter.sh`

### Docker Configuration
- **Image Name**: `umd_msml610/umd_msml610_image`
- **Container Name**: `umd_msml610_jupyter`
- **Default Port**: 8888
- **Volume Mount**: Current directory → `/data`

### Environment Variables

Create a `.env` file in the project root:
```bash
# Weights & Biases API Key
WANDB_API_KEY=your_wandb_api_key_here
```

## Project Structure

```
UmdTask50_Fall2025_Weights_&_Biases_Image_Classification_with_Transfer_Learning/
├── Dockerfile
├── docker_build.sh          # Build Docker image
├── docker_jupyter.sh        # Launch Jupyter in Docker
├── Weights_and_biases.example.ipynb            # Example/demo notebook
├── Weights_and_biases.train.ipynb              # Main training notebook (run on Google Colab with T4 GPU)
├── Weights_and_biases_utils.py                 # All utility functions (data prep, training, evaluation)
├── __init__.py              # Main training notebook (run on Google Colab with T4 GPU)
├── README.md                # This file is the main documentation of the project
└── Weights_and_biases.example.md # Exaplains the example/inference script of the project

```

### Key Files

**`Weights_and_biases_utils.py`**: Complete utility module containing:
- **Data Preparation Functions**:
  - `download_dataset()`: Downloads AFHQ dataset from Kaggle
  - `collect_image_dataframes()`: Creates train/validation DataFrames from directory structure
  - `create_image_data_generators()`: Applies data augmentation and creates Keras generators
  - `main_prep()`: Complete data preparation pipeline with verification
  
- **Model Building and Training Functions**:
  - `build_model()`: Constructs transfer learning models (ResNet50, EfficientNetB0, MobileNetV2)
  - `train_model()`: Trains individual models with W&B logging
  - `ensemble_models()`: Evaluates ensemble predictions using soft voting
  - `homogeneous_ensemble()`: Trains multiple instances of ResNet50
  - `heterogeneous_ensemble()`: Trains different architectures and combines predictions
  - `download_wandb_models_only()`: Downloads trained models from W&B artifacts

**`Weights_and_biases.train.ipynb`**: Main training notebook
- GPU configuration
- W&B authentication
- Complete training pipeline execution
- Single, homogeneous, and heterogeneous ensemble training
- **Note**: Optimized for Google Colab with T4 GPU; local training may not be efficient

## Usage

### Quick Start

1. **Setup W&B Account**
```bash
wandb login
# Or set WANDB_API_KEY in .env file
```

2. **Run Complete Training Pipeline**
   - Open `Weights_and_biases.train.ipynb` in Google Colab or Jupyter
   - Execute all cells to run the full pipeline
   - The notebook handles preprocessing, training, and logging automatically

## Model Training

### Training Pipeline

The `Weights_and_biases_utils.py` module provides a complete training pipeline:

1. **Data Preparation** (`main_prep()`)
   - Download AFHQ dataset from Kaggle
   - Create train/validation DataFrames from directory structure
   - Apply data augmentation
   - Generate TensorFlow data generators
   - Verify image counts and generator status

2. **Model Training** (`train_model()`)
   - Load pre-trained weights from ImageNet
   - Freeze base layers (except last N layers)
   - Add custom classification head
   - Train with validation monitoring
   - Log metrics to Weights & Biases

3. **Ensemble Creation**
   - `homogeneous_ensemble()`: Train multiple instances of ResNet50
   - `heterogeneous_ensemble()`: Train different architectures
   - Collect predictions on validation set
   - Apply soft voting (average predictions)
   - Compute and log ensemble accuracy

### Training Configuration
```python
# Default configuration in Weights_and_biases_utils.py
DATASET_REF = "andrewmvd/animal-faces"
BASE_SUBDIR = 'afhq'
CLASSES = ['cat', 'dog', 'wild']
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 256

# Configurable parameters
NUM_EPOCHS = 20
LEARNING_RATE = 0.00001
TRAINABLE_LAYERS = 50
```

### GPU Configuration

The project automatically detects and configures GPUs. Add this to your notebook:

**Note**: Training is optimized for Google Colab with T4 GPU. Local training may not be the most efficient use of resources.

## Evaluation

- Final validation loss and accuracy are logged in Weights & Biases
- Model checkpoints are saved as `model_checkpoint.keras`
- Ensemble predictions are evaluated on validation set using soft voting
- All metrics are tracked per-epoch and logged to W&B dashboard
- The `ensemble_models()` function computes ensemble accuracy by averaging predictions

## Results & Logging

### Weights & Biases Integration

All experiments are logged to W&B with:
- Training/validation loss and accuracy
- Model architecture details
- Hyperparameters
- Model checkpoints (via `WandbModelCheckpoint`)
- System metrics

### Metrics Tracked

- **Per-Epoch Metrics**:
  - Training accuracy
  - Training loss
  - Validation accuracy
  - Validation loss
  - Learning rate

- **Final Metrics**:
  - Individual model validation accuracy
  - Homogeneous ensemble accuracy
  - Heterogeneous ensemble accuracy

### W&B Project Structure

All models are logged to the `animal-faces-classification` project with unique run names:
- Single models: `"ResNet50"`, `"EfficientNetB0"`, `"MobileNetV2"`
- Homogeneous ensemble: `"ResNet50_Homo_1"`, `"ResNet50_Homo_2"`, `"ResNet50_Homo_3"`
- Heterogeneous ensemble: `"ResNet50_Hetero"`, `"EfficientNetB0_Hetero"`, `"MobileNetV2_Hetero"`
- Ensemble evaluations: `"homogeneous_ensemble"`, `"heterogeneous_ensemble"`

## Expected Performance

Based on the training logs:

| Model/Ensemble              | Validation Accuracy | # Parameters (approx.) |
|-----------------------------|------------------|----------------------|
| Single ResNet50             | 94.3%            | 25.6M                |
| Homogeneous Ensemble        | 94.1%            | 76.8M (3×ResNet50)  |
| Heterogeneous Ensemble      | 98.9%            | 63M (ResNet50 + EfficientNetB0 + MobileNetV2) |
| Single MobileNetV2          | 99.5%            | 3.5M                 |

## Why MobileNetV2 Performs Best

MobileNetV2 outperforms both single ResNet50/EfficientNetB0 models and the ensembles due to its architectural advantages:

1. **Inverted Residual Blocks** – Enables better feature propagation with fewer parameters.  
2. **Efficient Depthwise Separable Convolutions** – Extracts meaningful features efficiently, reducing overfitting on small datasets like `animal-faces`.  
3. **Lightweight yet Powerful** – Achieves **higher accuracy with ~7× fewer parameters** than ResNet50.  
4. **Fast Convergence** – Its design allows faster training and better generalization.  

> Despite ensembles having more parameters, MobileNetV2’s efficient design allows it to surpass them on this dataset.

## Configuration

### Model Hyperparameters
```python
# When calling training functions
import Weights_and_biases_utils as utils

model, acc = utils.train_model(
    train_gen, val_gen,
    architecture="ResNet50",    # Model architecture
    epochs=20,                   # Training epochs
    lr=0.00001,                  # Learning rate
    trainable_layers=50,         # Number of layers to fine-tune
    unique_name=None             # Optional unique name for W&B logging
)
```

### Data Processing Constants
```python
# In Weights_and_biases_utils.py
DATASET_REF = "andrewmvd/animal-faces"
BASE_SUBDIR = 'afhq'
CLASSES = ['cat', 'dog', 'wild']
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 256
num_workers = 16
```

### Augmentation Parameters
```python
# Training data augmentation (in Weights_and_biases_utils.py)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)
```

##  Notes

- `Weights_and_biases.train.ipynb` is currently used for training, not a demo/tutorial
- `Weights_and_biases.example.ipynb` can be used for demonstration purposes
- `Weights_and_biases_utils.py` contains all utility functions in a single module for simplicity
- Training on Google Colab with T4 GPU is recommended for efficiency

## Acknowledgments

- **Dataset**: AFHQ (Animal Faces-HQ) by andrewmvd on Kaggle
- **Frameworks**: TensorFlow/Keras, Weights & Biases
- **Pre-trained Models**: ImageNet weights
