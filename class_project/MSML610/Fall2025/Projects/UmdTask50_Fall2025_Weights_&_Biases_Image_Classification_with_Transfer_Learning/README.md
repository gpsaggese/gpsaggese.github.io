# Image Classification with Transfer Learning

## Overview
This project implements **transfer learning** to classify animal faces using a pre-trained **ResNet50** model.  
Training and evaluation metrics are tracked with **Weights & Biases (W&B)**.  

The project follows the **MSML610 Fall2025 class project template** and uses modular utility scripts for preprocessing and post-processing.


## Folder Structure

```text
UmdTask50_Fall2025_Weights_&_Biases_Image_Classification_with_Transfer_Learning/
├── Dockerfile
├── docker_build.sh
├── docker_bash.sh
├── docker_jupyter.sh
├── train.ipynb          # notebook for training the model
├── README.md
├── utils_preprocessing.py # data preparation functions
├── utils_post_processing.py # model training, evaluation, W&B integration
```

## Dataset
- **Animal Faces (AFHQ subset)** from Kaggle.  
- Preprocessing is handled via `utils_preprocessing.py`.  
- Training and validation generators are created with augmentation applied.


## Setup with Docker

All dependencies are installed inside the Docker image. No local `pip install` is required.

```bash
# Clone the repository
> git clone <repo-url>
> cd class_project/MSML610/Fall2025/Projects/UmdTask50_Fall2025_Weights_&_Biases_Image_Classification_with_Transfer_Learning

# Build the Docker image
> ./docker_build.sh

# Run interactive bash in the container
> ./docker_bash.sh

# Launch Jupyter Notebook inside the container
> ./docker_jupyter.sh
```

- The container mounts your project folder, so all notebooks and scripts are accessible.
- W&B metrics will work if .env with your WANDB_API_KEY is present in the project folder.
- You can map a different host port for Jupyter by passing -p <port> to docker_jupyter.sh.


## Training 
- The train.ipynb notebook is the main training script.
- It runs preprocessing, trains the model, and logs metrics to Weights & Biases.
- Model checkpoints are saved as model_checkpoint.keras.


## Evaluation 
- Final validation loss and accuracy are logged in W&B.
- Model checkpoints are saved as model_checkpoint.keras.


## Notes

- train.ipynb is currently used for training, not a demo/tutorial.
- Utilities (utils_preprocessing.py and utils_post_processing.py) separate data preparation, training, and logging.
