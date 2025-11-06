# Image Classification with Transfer Learning

## Overview
This project uses **transfer learning** to classify animal faces using a pre-trained **ResNet50** model.  
Training and evaluation metrics are tracked with **Weights & Biases (W&B)**.

## Dataset
- **Animal Faces** dataset (AFHQ subset) from Kaggle.
- Data is prepared using `dataset_operations.main_prep()`.
- Training and validation generators are created with augmentation applied.

## Setup
```bash
# Clone repo
> git clone <repo-url>
> cd msml610/tutorials/Weights_and_Biases

# Install dependencies
> pip install -r requirements.txt

# Create .env with your WANDB_API_KEY
> echo "WANDB_API_KEY=<your_key>" > .env
```

## Training 

```bash 
> python train_resnet.py
```

## Evaluation 
- Final validation loss and accuracy are logged in W&B.
- Model checkpoints are saved as model_checkpoint.keras.