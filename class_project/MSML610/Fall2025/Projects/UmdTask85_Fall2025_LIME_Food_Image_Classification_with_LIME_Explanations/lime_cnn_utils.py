"""
Util functions for:
- Data loading and preprocessing
- CNN model construction and training
- LIME explanation generation
- Model evaluation and visualization
"""

import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from lime import lime_image
from skimage.segmentation import mark_boundaries


# Custom dataset class for fast loading from metadata
class FastFoodDataset(Dataset):
    """Custom dataset class for loading Food-101 images from selected paths."""
    def __init__(self, image_paths_and_labels, transform=None, classes=None, class_to_idx=None):
        self.image_paths_and_labels = image_paths_and_labels
        self.transform = transform
        self.classes = classes if classes is not None else []
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}
        # Store image paths for compatibility with other functions
        self._image_files = [path for path, _ in image_paths_and_labels]
    
    def __len__(self):
        return len(self.image_paths_and_labels)
    
    def __getitem__(self, idx):
        img_path, label = self.image_paths_and_labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# data utils

def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transformation pipelines for training and validation.
    
    Training transforms include comprehensive data augmentation:
    - Resize and center crop for consistent input size
    - Random horizontal flip
    - Random rotation
    - Color jitter (brightness, contrast, saturation)
    - Random affine transformations
    - Normalization using ImageNet statistics
    
    Validation transforms include only:
    - Resize and center crop
    - Normalization using ImageNet statistics
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to slightly larger size
        transforms.RandomCrop(224),  # Random crop for augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),  # Center crop for validation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return train_transform, val_transform


def get_test_image_path(test_dataset, test_dataset_full, idx):
    """Get test image path by index."""
    if isinstance(test_dataset, Subset):
        actual_idx = test_dataset.indices[idx]
        return test_dataset_full._image_files[actual_idx]
    else:
        return test_dataset_full._image_files[idx]

def get_test_image_label(test_dataset, test_dataset_full, idx):
    """Get test image label by index."""
    if isinstance(test_dataset, Subset):
        actual_idx = test_dataset.indices[idx]
        image, label = test_dataset_full[actual_idx]
    else:
        image, label = test_dataset_full[idx]
    return label


def create_balanced_subset_from_metadata(
    metadata_path, 
    data_root, 
    all_class_names, 
    total_samples, 
    transform,
    selected_classes=None,
    num_classes_to_use=None,
    random_seed=42
):
    """
    Create a balanced subset dataset directly from JSON metadata files.
    This is much faster than iterating through the entire dataset.
    
    Args:
        metadata_path: Path to JSON metadata file (train.json or test.json)
        data_root: Root directory of the dataset
        all_class_names: List of all available class names (ordered)
        total_samples: Total number of samples to select
        transform: Transform to apply to images
        selected_classes: Optional list of specific class names to use. If None, uses all classes.
        num_classes_to_use: Optional number of classes to randomly select. If None, uses all or selected_classes.
        random_seed: Random seed for reproducibility
    
    Returns:
        Custom dataset with selected images and updated class mapping
    """
    import json
    import random as random_module
    
    random_module.seed(random_seed)
    
    # Load metadata JSON
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Determine which classes to use
    if selected_classes is not None:
        # Use specified classes
        class_names_to_use = [c for c in selected_classes if c in metadata]
        if len(class_names_to_use) == 0:
            raise ValueError(f"None of the selected classes found in metadata: {selected_classes}")
    elif num_classes_to_use is not None and num_classes_to_use < len(all_class_names):
        # Randomly select N classes
        available_classes = [c for c in all_class_names if c in metadata]
        if num_classes_to_use > len(available_classes):
            raise ValueError(f"Requested {num_classes_to_use} classes but only {len(available_classes)} available")
        class_names_to_use = sorted(random_module.sample(available_classes, num_classes_to_use))
    else:
        # Use all classes
        class_names_to_use = [c for c in all_class_names if c in metadata]
    
    # Create mapping from original class index to new class index
    class_to_idx_new = {name: idx for idx, name in enumerate(class_names_to_use)}
    
    # Calculate samples per class
    num_classes = len(class_names_to_use)
    samples_per_class = total_samples // num_classes
    remainder = total_samples % num_classes
    
    # Select images evenly from each selected class
    selected_images = []  # List of (image_path, new_class_idx) tuples
    
    for class_idx, class_name in enumerate(class_names_to_use):
        if class_name not in metadata:
            continue
        
        # Get available images for this class
        class_images = metadata[class_name]
        if len(class_images) == 0:
            continue
        
        # Calculate how many to select from this class
        num_to_select = samples_per_class + (1 if class_idx < remainder else 0)
        num_to_select = min(num_to_select, len(class_images))
        
        # Randomly sample from this class
        selected_paths = random_module.sample(class_images, num_to_select)
        
        # Add full paths with new class index
        for img_path in selected_paths:
            # img_path is like "churros/1004234", need to add .jpg and full path
            full_path = Path(data_root) / "images" / f"{img_path}.jpg"
            selected_images.append((str(full_path), class_idx))
    
    # Shuffle to mix classes
    random_module.shuffle(selected_images)
    
    # Create dataset using the module-level class (pickleable)
    return FastFoodDataset(
        image_paths_and_labels=selected_images, 
        transform=transform, 
        classes=class_names_to_use,
        class_to_idx=class_to_idx_new
    )


def create_balanced_subset(dataset, total_samples, class_names, random_seed=42):
    """
    Create a balanced subset of the dataset with equal number of samples per class.
    NOTE: This is slower than create_balanced_subset_from_metadata. Use that instead when possible.
    
    Args:
        dataset: PyTorch dataset
        total_samples: Total number of samples to select
        class_names: List of class names
        random_seed: Random seed for reproducibility
    
    Returns:
        Subset of the dataset with balanced class distribution
    """
    import random as random_module
    random_module.seed(random_seed)
    
    # Group indices by class
    indices_by_class = {i: [] for i in range(len(class_names))}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        indices_by_class[label].append(idx)
    
    # Calculate samples per class
    num_classes = len(class_names)
    samples_per_class = total_samples // num_classes
    remainder = total_samples % num_classes
    
    # Select indices evenly from each class
    selected_indices = []
    for class_idx in range(num_classes):
        class_indices = indices_by_class[class_idx]
        if len(class_indices) == 0:
            continue
        
        # Add one extra sample to first 'remainder' classes to use up all samples
        num_to_select = samples_per_class + (1 if class_idx < remainder else 0)
        num_to_select = min(num_to_select, len(class_indices))
        
        # Randomly sample from this class
        selected = random_module.sample(class_indices, num_to_select)
        selected_indices.extend(selected)
    
    # Shuffle to mix classes
    random_module.shuffle(selected_indices)
    
    return Subset(dataset, selected_indices)


def count_images_per_class(dataset, class_names):
    """
    Count the number of images per class in a dataset.
    
    Args:
        dataset: PyTorch dataset (can be Subset or full dataset)
        class_names: List of class names
    
    Returns:
        Dictionary mapping class names to counts
    """
    from collections import Counter
    
    counts = Counter()
    
    for idx in range(len(dataset)):
        if isinstance(dataset, Subset):
            _, label = dataset.dataset[dataset.indices[idx]]
        else:
            _, label = dataset[idx]
        
        class_name = class_names[label]
        counts[class_name] += 1
    
    return dict(counts)


def plot_images_per_class(train_dataset, test_dataset, class_names, save_path=None):
    """
    Plot the distribution of images per class for train and test datasets.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    train_counts = count_images_per_class(train_dataset, class_names)
    test_counts = count_images_per_class(test_dataset, class_names)
    
    # Sort classes by name for consistent ordering
    sorted_classes = sorted(class_names)
    train_values = [train_counts.get(cls, 0) for cls in sorted_classes]
    test_values = [test_counts.get(cls, 0) for cls in sorted_classes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot training distribution
    x_pos = np.arange(len(sorted_classes))
    width = 0.35
    
    axes[0].bar(x_pos, train_values, width, label='Train', color='#1f77b4', alpha=0.8)
    axes[0].set_xlabel('Class', fontsize=10)
    axes[0].set_ylabel('Number of Images', fontsize=10)
    axes[0].set_title(f'Training Set: Images per Class (Total: {sum(train_values)} images)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(sorted_classes, rotation=90, ha='right', fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot test distribution
    axes[1].bar(x_pos, test_values, width, label='Test', color='#ff7f0e', alpha=0.8)
    axes[1].set_xlabel('Class', fontsize=10)
    axes[1].set_ylabel('Number of Images', fontsize=10)
    axes[1].set_title(f'Test Set: Images per Class (Total: {sum(test_values)} images)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(sorted_classes, rotation=90, ha='right', fontsize=8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# model utils

def create_cnn_model(
    num_classes: int, architecture: str = 'resnet18', pretrained: bool = True
) -> nn.Module:
    """
    Create a CNN model for food classification.
    
    Supports multiple architectures:
    - resnet18: Fast, lightweight ResNet with 18 layers
    - resnet50: Deeper ResNet with 50 layers for better accuracy
    - efficientnet_b0: EfficientNet-B0 for best accuracy/speed tradeoff
    
    Args:
        num_classes: Number of food classes
        architecture: Model architecture ('resnet18', 'resnet50', 'efficientnet_b0')
        pretrained: Whether to use ImageNet pretrained weights (recommended)
    
    Returns:
        PyTorch model ready for training
    """
    if architecture == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'efficientnet_b0' or architecture == 'efficientnet':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose from: 'resnet18', 'resnet50', 'efficientnet_b0'")
    
    return model

def save_model(model: nn.Module, path: str, class_to_idx: dict):
    """
    Save model and metadata.
    
    Args:
        model: Trained PyTorch model
        path: Path to save the model
        class_to_idx: Class to index mapping
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, model: nn.Module, device: str = 'cuda') -> dict:
    """
    Load model and metadata.
    
    Args:
        path: Path to the saved model
        model: Model architecture to load weights into
        device: Device to load model on
    
    Returns:
        Dictionary with model and metadata
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return {
        'model': model,
        'class_to_idx': checkpoint.get('class_to_idx', {})
    }


# train utils

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    num_epochs: int = 10,
    device: str = 'cuda', 
    learning_rate: float = 0.001,
    log_dir: Optional[str] = None
) -> List[dict]:
    """
    Train the CNN model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        learning_rate: Learning rate for optimizer
        log_dir: Optional directory for TensorBoard logs. If None, logging is disabled.
    
    Returns:
        List of training history dictionaries
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    writer = None
    if log_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"run_{timestamp}"
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_path))
    
    history = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training phase with progress bar
        train_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=True,
            ncols=100
        )
        
        for batch_idx, (images, labels) in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_acc = 100 * train_correct / train_total if train_total > 0 else 0
            current_loss = train_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            # log batch-level metrics
            if writer is not None and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                writer.add_scalar('Train/BatchAccuracy', batch_acc, global_step)
        
        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
            leave=True,
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, (images, labels) in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar with current metrics
                current_acc = 100 * val_correct / val_total if val_total > 0 else 0
                current_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'epoch_time': epoch_time,
            'learning_rate': current_lr
        }
        history.append(epoch_history)
        
        # log epoch-level metrics
        if writer is not None:
            writer.add_scalar('Metrics/TrainLoss', avg_train_loss, epoch + 1)
            writer.add_scalar('Metrics/TrainAccuracy', train_acc, epoch + 1)
            writer.add_scalar('Metrics/ValLoss', avg_val_loss, epoch + 1)
            writer.add_scalar('Metrics/ValAccuracy', val_acc, epoch + 1)
            writer.add_scalar('LearningRate', current_lr, epoch + 1)
            writer.add_scalar('Time/EpochTime', epoch_time, epoch + 1)
        
        scheduler.step()
    
    if writer is not None:
        writer.close()
    
    return history


# eval utils

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cuda') -> dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    eval_pbar = tqdm(
        test_loader,
        total=len(test_loader),
        desc="Evaluating",
        leave=True,
        ncols=100
    )
    
    with torch.no_grad():
        for images, labels in eval_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with current accuracy
            current_acc = 100 * correct / total if total > 0 else 0
            eval_pbar.set_postfix({
                'acc': f'{current_acc:.2f}%',
                'correct': f'{correct}/{total}'
            })
    
    accuracy = 100 * correct / total
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': all_preds,
        'labels': all_labels
    }


# LIME utils

def batch_predict(images: np.ndarray, model: nn.Module, device: str = 'cuda') -> np.ndarray:
    """
    Batch prediction function for LIME.
    
    Args:
        images: Batch of images as numpy array
        model: Trained PyTorch model
        device: Device to run inference on
    
    Returns:
        Predictions as numpy array
    """
    model.eval()
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    batch = batch.to(device)
    
    # normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    batch = normalize(batch)
    
    with torch.no_grad():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    
    return probs.cpu().numpy()

def explain_prediction(
    image_path: str, 
    model: nn.Module, 
    class_names: List[str], 
    device: str = 'cuda',
    num_features: int = 10, 
    num_samples: int = 1000,
    top_labels: int = 5
) -> dict:
    """
    Generate LIME explanation for a single image prediction.
    
    Args:
        image_path: Path to the image file
        model: Trained PyTorch model
        class_names: List of class names
        device: Device to run inference on
        num_features: Number of superpixels to highlight
        num_samples: Number of samples for LIME
        top_labels: Number of top labels to explain
    
    Returns:
        Dictionary containing explanation results
    """

    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # generate LIME explanation

    explainer = lime_image.LimeImageExplainer()
    def predict_fn(images):
        return batch_predict(images, model, device)
    
    explanation = explainer.explain_instance(
        image_array,
        predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples
    )
    
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=num_features,
        hide_rest=False
    )
    
    # create visualization

    img_boundary = mark_boundaries(temp / 255.0, mask)
    
    # get prediction probabilities

    prediction_fn = lambda x: batch_predict(x, model, device)
    probs = prediction_fn(np.expand_dims(image_array, axis=0))[0]
    top_probs = sorted(
        enumerate(probs), key=lambda x: x[1], reverse=True
    )[:top_labels]
    
    return {
        'image': image_array,
        'mask': mask,
        'visualization': img_boundary,
        'top_label': top_label,
        'top_label_name': class_names[top_label],
        'top_probabilities': [(class_names[idx], float(prob)) for idx, prob in top_probs],
        'explanation': explanation
    }


def visualize_explanation(explanation_result: dict, save_path: Optional[str] = None):
    """
    Visualize LIME explanation results with highlighted regions.
    
    Shows:
    - Original image
    - LIME explanation with highlighted superpixels
    - Top predictions bar chart
    
    Args:
        explanation_result: Result dictionary from explain_prediction
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # original image
    axes[0].imshow(explanation_result['image'])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # explanation with highlighted regions
    axes[1].imshow(explanation_result['visualization'])
    axes[1].set_title(f'LIME Explanation\nPredicted: {explanation_result["top_label_name"]}')
    axes[1].axis('off')
    
    # top predictions
    classes, probs = zip(*explanation_result['top_probabilities'])
    axes[2].barh(range(len(classes)), probs)
    axes[2].set_yticks(range(len(classes)))
    axes[2].set_yticklabels(classes)
    axes[2].set_xlabel('Probability')
    axes[2].set_title('Top Predictions')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_explanation_detailed(explanation_result: dict, save_path: Optional[str] = None):
    """
    Visualize LIME explanation with detailed positive/negative contributions.
    
    Shows:
    - Original image
    - Positive contributions (regions supporting prediction)
    - Negative contributions (regions against prediction)
    - Combined explanation
    
    Args:
        explanation_result: Result dictionary from explain_prediction
        save_path: Optional path to save the visualization
    """
    explanation = explanation_result['explanation']
    top_label = explanation_result['top_label']
    image = explanation_result['image']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Positive contributions
    temp_pos, mask_pos = explanation.get_image_and_mask(
        top_label, positive_only=True, negative_only=False, 
        num_features=5, hide_rest=True
    )
    img_boundary_pos = mark_boundaries(temp_pos / 255.0, mask_pos)
    axes[0, 1].imshow(img_boundary_pos)
    axes[0, 1].set_title('Positive Contributions\n(Supporting Prediction)')
    axes[0, 1].axis('off')
    
    # Negative contributions
    temp_neg, mask_neg = explanation.get_image_and_mask(
        top_label, positive_only=False, negative_only=True, 
        num_features=5, hide_rest=True
    )
    img_boundary_neg = mark_boundaries(temp_neg / 255.0, mask_neg)
    axes[1, 0].imshow(img_boundary_neg)
    axes[1, 0].set_title('Negative Contributions\n(Against Prediction)')
    axes[1, 0].axis('off')
    
    # Combined explanation
    axes[1, 1].imshow(explanation_result['visualization'])
    axes[1, 1].set_title(f'Combined Explanation\nPredicted: {explanation_result["top_label_name"]}')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_architectures(
    architectures: List[str],
    test_loader: DataLoader,
    num_classes: int,
    model_dir: str = 'models',
    device: str = 'cuda'
) -> dict:
    """
    Compare multiple CNN architectures by loading pretrained models.
    
    Args:
        architectures: List of architecture names to compare
        test_loader: DataLoader for test data
        num_classes: Number of classes
        model_dir: Directory containing saved models
        device: Device to load models on
    
    Returns:
        Dictionary with comparison results for each architecture
    """
    results = {}
    model_dir_path = Path(model_dir)
    
    for arch in architectures:
        model_path = model_dir_path / f"food_classifier_{arch}.pth"
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print(f"Skipping {arch}. Please train the model first.")
            continue
        
        print(f"\nLoading {arch}")
        
        # Create model architecture
        model = create_cnn_model(num_classes=num_classes, architecture=arch, pretrained=False)
        
        # Load pretrained weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Evaluate on test set
        test_results = evaluate_model(model, test_loader, device=device)
        
        results[arch] = {
            'model': model,
            'test_accuracy': test_results['accuracy'],
            'test_correct': test_results['correct'],
            'test_total': test_results['total'],
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_path': str(model_path)
        }
        
        print(f"{arch} Test Accuracy: {test_results['accuracy']:.2f}%")
    
    return results