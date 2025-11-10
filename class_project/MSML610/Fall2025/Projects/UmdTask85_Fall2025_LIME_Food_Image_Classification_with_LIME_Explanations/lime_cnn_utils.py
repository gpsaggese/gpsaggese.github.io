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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

from PIL import Image
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries


# data utils

def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transformation pipelines for training and validation.
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_test_image_path(idx):
    """Get test image path by index."""
    if isinstance(test_dataset, Subset):
        actual_idx = test_dataset.indices[idx]
        return test_dataset_full._image_files[actual_idx]
    else:
        return test_dataset_full._image_files[idx]

def get_test_image_label(idx):
    """Get test image label by index."""
    if isinstance(test_dataset, Subset):
        actual_idx = test_dataset.indices[idx]
        image, label = test_dataset_full[actual_idx]
    else:
        image, label = test_dataset_full[idx]
    return label


# model utils

def create_cnn_model(
    num_classes: int, architecture: str = 'resnet18', pretrained: bool = True
) -> nn.Module:
    """
    Create a CNN model for food classification.
    
    Args:
        num_classes: Number of food classes
        architecture: Model architecture ('resnet18', 'resnet50', 'efficientnet')
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
    if architecture == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
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

        print(f"TensorBoard logging enabled. Logs saved to: {log_path}")
        print(f"View logs with: tensorboard --logdir={log_path}")
    
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            # log batch-level metrics
            if writer is not None and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                writer.add_scalar('Train/BatchAccuracy', batch_acc, global_step)
        
        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc
        }
        history.append(epoch_history)
        
        # log epoch-level metrics
        if writer is not None:
            writer.add_scalar('Metrics/TrainLoss', epoch_history['train_loss'], epoch + 1)
            writer.add_scalar('Metrics/TrainAccuracy', train_acc, epoch + 1)
            writer.add_scalar('Metrics/ValLoss', epoch_history['val_loss'], epoch + 1)
            writer.add_scalar('Metrics/ValAccuracy', val_acc, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_history["train_loss"]:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {epoch_history["val_loss"]:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        scheduler.step()
    
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {log_path}")
    
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
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    Visualize LIME explanation results.
    
    Args:
        explanation_result: Result dictionary from explain_prediction
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # original image
    axes[0].imshow(explanation_result['image'])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # explanation
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