# Animal Faces Classification - Ensemble Inference Example

This example demonstrates ensemble inference using multiple trained models from Weights & Biases for classifying animal faces into three categories: **cat**, **dog**, and **wild**.

## Overview

The code performs **soft voting ensemble inference**:
1. Downloads trained models from W&B
2. Loads sample images for each class
3. Runs predictions across all models
4. Averages predictions and displays results

## Configuration

```python
IMG_HEIGHT, IMG_WIDTH = 128, 128
ENTITY = "pshashid-university-of-maryland"
PROJECT = "animal-faces-classification"
CLASS_INDICES = {'cat': 0, 'dog': 1, 'wild': 2}
RANDOM_STATE = 40
```

### Models Used

**Heterogeneous Models** (grayscale input):
- `run_mobilenetV2_hetero`
- `run_efficientnetb0_hetero`
- `run_resnet50_hetero`

**Homogeneous Models** (RGB input):
- `run_ResNet50_Homo_1/2/3`
- `run_resnet50`

## How It Works

### Soft Voting

Predictions are averaged across all models:

```
Model 1: [0.2, 0.7, 0.1]
Model 2: [0.1, 0.8, 0.1]
Model 3: [0.3, 0.6, 0.1]
         ----------------
Average: [0.2, 0.7, 0.1] → Predicts "dog"
```

## Expected Output

For each test image:
```
Ensemble Prediction
True: cat, Predicted: cat
```

With image visualization displayed.


## Performance Notes

- **Memory**: ~2-3GB for 7 models
- **Speed**: 7× slower than single model inference
- **Accuracy**: Ensemble may not always improve results, test against individual models

## Note on Ensemble Performance

Ensemble methods don't guarantee better accuracy. If individual models have similar biases or if they're not diverse enough, ensembling can actually degrade performance. Consider:
- Using only the best individual model
- Testing weighted voting based on validation accuracy
- Ensuring model diversity (different architectures, training strategies)