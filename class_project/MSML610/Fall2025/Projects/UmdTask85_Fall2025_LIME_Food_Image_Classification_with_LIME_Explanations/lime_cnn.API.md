# LIME CNN API Documentation

This document describes the native programming interfaces for LIME (Local Interpretable Model-agnostic Explanations) and PyTorch CNN models, along with the lightweight wrapper layer provided in `lime_cnn_utils.py`.

## Overview

The project integrates two main technologies:
1. **PyTorch/Torchvision**: For building and training CNN models for food image classification
2. **LIME**: For generating interpretable explanations of CNN predictions

## Native LIME API

LIME provides model-agnostic explanations through the `lime_image` module. The core interface is:

### `LimeImageExplainer`

```python
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
```

**Key Methods:**

- `explain_instance(image, classifier_fn, top_labels=5, hide_color=0, num_samples=1000)`
  - `image`: Input image as numpy array
  - `classifier_fn`: Function that takes a batch of images and returns predictions
  - `top_labels`: Number of top classes to explain
  - `hide_color`: Color to use when hiding superpixels
  - `num_samples`: Number of perturbed samples to generate
  - Returns: `Explanation` object

**Explanation Object Methods:**

- `get_image_and_mask(label, positive_only=True, num_features=10, hide_rest=False)`
  - Returns the image and mask highlighting important regions
  - `label`: Class index to explain
  - `positive_only`: Show only positive contributions
  - `num_features`: Number of superpixels to highlight
  - `hide_rest`: Hide non-important regions

## Native PyTorch/Torchvision API

### Model Architectures

PyTorch's `torchvision.models` provides pretrained CNN architectures:

```python
from torchvision import models

# ResNet architectures
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# EfficientNet
efficientnet = models.efficientnet_b0(pretrained=True)
```

### Data Transforms

`torchvision.transforms` provides image preprocessing:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

dataset = CustomDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Wrapper Layer API

The `lime_cnn_utils.py` module provides a higher-level interface that simplifies common workflows.

### Data Management

#### `FastFoodDataset`

A custom PyTorch Dataset class for efficient loading of Food-101 images from metadata:

```python
from lime_cnn_utils import FastFoodDataset

dataset = FastFoodDataset(
    image_paths_and_labels=[(path1, label1), (path2, label2), ...],
    transform=train_transform,
    classes=class_names,
    class_to_idx=class_to_idx_mapping
)
```

**Attributes:**
- `classes`: List of class names
- `class_to_idx`: Dictionary mapping class names to indices
- `_image_files`: List of image file paths (for compatibility with other functions)

**Note:** This class is designed for efficient loading when using metadata-driven subset creation.

#### `create_balanced_subset_from_metadata()`

Creates a balanced subset dataset directly from JSON metadata files (much faster than iterating through the full dataset):

```python
from lime_cnn_utils import create_balanced_subset_from_metadata

dataset = create_balanced_subset_from_metadata(
    metadata_path="data/food-101/meta/train.json",
    data_root="data",
    all_class_names=all_classes,
    total_samples=1000,
    transform=train_transform,
    selected_classes=None,  # Optional: specific classes to use
    num_classes_to_use=5,  # Optional: number of classes to randomly select
    random_seed=42
)
```

**Key Features:**
- Loads directly from JSON metadata (no need to iterate through full dataset)
- Supports class selection (specific classes or random N classes)
- Ensures balanced distribution across selected classes
- Much faster than traditional subset creation methods

#### `get_data_transforms()`

Returns standardized transform pipelines:

```python
train_transform, val_transform = get_data_transforms()
```

### Model Construction

#### `create_cnn_model(num_classes, architecture='resnet18', pretrained=True)`

Creates a CNN model with customizable architecture:

```python
model = create_cnn_model(
    num_classes=101,
    architecture='resnet50',  # Options: 'resnet18', 'resnet50', 'efficientnet'
    pretrained=True
)
```

**Architecture Support:**
- `resnet18`: Lightweight ResNet (18 layers)
- `resnet50`: Deeper ResNet (50 layers)
- `efficientnet`: EfficientNet-B0 architecture

### Training

#### `train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', learning_rate=0.001, log_dir=None)`

Trains the model with automatic validation and progress tracking:

```python
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    device='cuda',
    learning_rate=0.001,
    log_dir="runs"  # Optional: TensorBoard logging directory
)
```

**Features:**
- Real-time progress bars using `tqdm` for training and validation phases
- Displays current loss and accuracy metrics during training
- Optional TensorBoard logging for detailed metric tracking
- Automatic learning rate scheduling

**Returns:** List of dictionaries containing epoch-wise metrics:
- `epoch`: Epoch number
- `train_loss`: Training loss
- `train_acc`: Training accuracy (%)
- `val_loss`: Validation loss
- `val_acc`: Validation accuracy (%)
- `epoch_time`: Time taken for the epoch
- `learning_rate`: Learning rate at epoch end

### Evaluation

#### `evaluate_model(model, test_loader, device='cuda')`

Evaluates model performance with progress tracking:

```python
results = evaluate_model(model, test_loader, device='cuda')
# Returns: {'accuracy': 85.5, 'correct': 855, 'total': 1000, 
#           'predictions': [...], 'labels': [...]}
```

**Features:**
- Real-time progress bar showing evaluation progress
- Displays current accuracy and correct/total counts
- Collects all predictions and labels for detailed analysis

### LIME Integration

#### `batch_predict(images, model, device='cuda')`

Wrapper function for LIME's prediction interface:

```python
predictions = batch_predict(image_batch, model, device='cuda')
```

This function handles:
- Image normalization (ImageNet stats)
- Tensor conversion and device placement
- Softmax probability computation

#### `explain_prediction(image_path, model, class_names, device='cuda', num_features=10, num_samples=1000, top_labels=5)`

High-level function for generating explanations:

```python
explanation = explain_prediction(
    image_path="data/test_image.jpg",
    model=model,
    class_names=class_names,
    device='cuda',
    num_features=10,
    num_samples=1000,
    top_labels=5
)
```

**Returns:** Dictionary containing:
- `image`: Original image array
- `mask`: Superpixel mask
- `visualization`: Image with boundaries marked
- `top_label`: Predicted class index
- `top_label_name`: Predicted class name
- `top_probabilities`: List of (class_name, probability) tuples
- `explanation`: Raw LIME Explanation object

#### `visualize_explanation(explanation_result, save_path=None)`

Creates a comprehensive visualization:

```python
visualize_explanation(explanation, save_path="output/explanation.png")
```

Displays:
1. Original image
2. LIME explanation overlay
3. Top prediction probabilities

### Model Persistence

#### `save_model(model, path, class_to_idx)`

Saves model and metadata:

```python
save_model(model, "models/food_classifier.pth", dataset.class_to_idx)
```

#### `load_model(path, model, device='cuda')`

Loads saved model:

```python
checkpoint = load_model("models/food_classifier.pth", model, device='cuda')
model = checkpoint['model']
class_to_idx = checkpoint['class_to_idx']
```

#### `compare_architectures(architectures, test_loader, num_classes, model_dir='models', device='cuda')`

Compares multiple CNN architectures by loading pretrained models:

```python
results = compare_architectures(
    architectures=['resnet18', 'resnet50', 'efficientnet_b0'],
    test_loader=test_loader,
    num_classes=101,
    model_dir='models',
    device='cuda'
)
```

## Integration Pattern

The wrapper layer follows a common pattern:

1. **Setup**: Load data, create model, configure transforms
2. **Training**: Train model with validation monitoring
3. **Evaluation**: Assess model performance
4. **Explanation**: Generate LIME explanations for selected images
5. **Visualization**: Display results in interpretable format

This pattern is demonstrated in the API and example notebook.