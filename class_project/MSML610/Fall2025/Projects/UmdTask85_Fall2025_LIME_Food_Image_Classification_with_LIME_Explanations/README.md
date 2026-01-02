# Food Image Classification with LIME Explanations

This project implements a food image classification system using Convolutional Neural Networks (CNNs) and Local Interpretable Model-agnostic Explanations (LIME) to explain model predictions.

Difficulty: Hard

Authors:

Damith Imbulana Liyanage (UID: 117489323)
Kumar Vaibhav (UID: 121092496)

## Project Overview

The goal is to:
1. Train a CNN to classify food images from the Food-101 dataset
2. Use LIME to explain which image regions influence the model's predictions
3. Compare different CNN architectures (ResNet-18, ResNet-50, EfficientNet) and their interpretability

## Features

- **Multiple CNN Architectures**: Support for ResNet-18, ResNet-50, and EfficientNet-B0
- **Transfer Learning**: Optionally uses ImageNet pretrained weights
- **LIME Integration**: Generate interpretable explanations for model predictions
- **Data Augmentation**: Random flips, rotations, and color jitter for better generalization
- **Visualization**: Visualize explanations with highlighted important regions

## Project Structure

```
.
├── lime_cnn_utils.py          # Utility functions and wrapper layer
├── lime_cnn.API.md            # API documentation
├── lime_cnn.API.ipynb         # API demonstration notebook
├── lime_cnn.example.md        # Complete example documentation
├── lime_cnn.example.ipynb     # End-to-end example notebook
├── Dockerfile                 # Docker configuration
├── docker_build.sh            # Build Docker image
├── docker_bash.sh             # Run container with bash
├── docker_jupyter.sh          # Run container with Jupyter
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/                      # Dataset directory
```

## Setup

#### Build the Image

```bash
chmod +x docker_build.sh
./docker_build.sh
```

#### Run the Container

For Jupyter Notebook:

```bash
chmod +x docker_jupyter.sh
./docker_jupyter.sh
```

Then open the shown link in your browser.

For Bash Access:

```bash
chmod +x docker_bash.sh
./docker_bash.sh
```

## Dataset

This project uses the Food-101 dataset which contains:
- 101 food categories
- 101,000 images total

### Dataset Structure

The dataset is organized as follows:

```
data/food101/
├── train/
│   ├── apple_pie/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── baby_back_ribs/
│   └── ...
└── test/
    ├── apple_pie/
    ├── baby_back_ribs/
    └── ...
```

[^1] Ribeiro, M., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939778.

[^2]: Bossard, L., Guillaumin, M., & Gool, L.V. (2014). Food-101 - Mining Discriminative Components with Random Forests. European Conference on Computer Vision. https://doi.org/10.1007/978-3-319-10599-4_29.