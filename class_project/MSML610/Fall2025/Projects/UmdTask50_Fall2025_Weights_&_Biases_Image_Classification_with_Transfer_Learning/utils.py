import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.applications
import tensorflow.keras.optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
import wandb.integration.keras
import kagglehub
import IPython.display as disp


# ----------------------------------------------------------------------
# --- Constants for Data Configuration ---
# ----------------------------------------------------------------------
DATASET_REF = "andrewmvd/animal-faces"
BASE_SUBDIR = 'afhq'
CLASSES = ['cat', 'dog', 'wild']
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32


# ----------------------------------------------------------------------
# --- Data Preparation Functions ---
# ----------------------------------------------------------------------

def download_dataset(dataset_ref: str = DATASET_REF) -> str:
    """
    Downloads the specified dataset from KaggleHub.

    Returns:
        The local file path to the downloaded dataset.
    """
    print(f"Downloading dataset: {dataset_ref}...")
    try:
        path = kagglehub.dataset_download(dataset_ref)
        print("Path to dataset files:", path)
        return path
    except Exception as e:
        print(f"An error occurred during dataset download: {e}")
        return ""


def collect_image_dataframes(dataset_path: str, base_subdir: str = BASE_SUBDIR, classes: list = CLASSES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data and performs the Train/Validation Split by collecting image paths
    based on the dataset's directory structure (afhq/train/* and afhq/val/*).

    Returns:
        A tuple containing (train_df, val_df) with 'image_path' and 'label' columns.
    """
    if not dataset_path:
        return pd.DataFrame(), pd.DataFrame()

    train_data = {'image_path': [], 'label': []}
    val_data = {'image_path': [], 'label': []}

    print("\nCollecting image paths and labels (performing Train/Validation Split)...")
    for class_name in classes:
        train_dir = os.path.join(dataset_path, base_subdir, 'train', class_name)
        val_dir = os.path.join(dataset_path, base_subdir, 'val', class_name)

        # Collect train data
        if os.path.exists(train_dir):
            for filename in os.listdir(train_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')): 
                    train_data['image_path'].append(os.path.join(train_dir, filename))
                    train_data['label'].append(class_name)

        # Collect validation data
        if os.path.exists(val_dir):
            for filename in os.listdir(val_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    val_data['image_path'].append(os.path.join(val_dir, filename))
                    val_data['label'].append(class_name)

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    print(f"\nTrain data collected. Samples: {len(train_df)}")
    disp.display(train_df.head())
    print(f"\nValidation data collected. Samples: {len(val_df)}")
    disp.display(val_df.head())

    return train_df, val_df


def create_image_data_generators(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                 img_height: int = IMG_HEIGHT, img_width: int = IMG_WIDTH, 
                                 batch_size: int = BATCH_SIZE) -> tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Performs Data Augmentation and creates Keras ImageDataGenerators.

    Returns:
        A tuple containing (train_generator, val_generator).
    """
    print("\nSetting up ImageDataGenerators (including Data Augmentation)...")

    # Data Augmentation for Training Set
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalization (mandatory)
        shear_range=0.2,         
        zoom_range=0.2,          
        horizontal_flip=True     # Augmentation applied only to training data
    )

    # Simple Rescaling for Validation Set (No Augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255) 

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True 
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("Data Augmentation applied. Generators created successfully.")
    return train_generator, val_generator


def main_prep():
    """
    Executes the data preparation pipeline: download, split, and augment,
    and includes verification of image counts.
    """
    print("--- Starting Data Preparation Pipeline ---")
    
    # 1. Download the dataset
    dataset_path = download_dataset()

    if not dataset_path:
        print("Exiting due to dataset download failure.")
        return

    # 2. Split data into DataFrames (Train/Validation)
    train_df, val_df = collect_image_dataframes(dataset_path)

    if train_df.empty or val_df.empty:
        print("Exiting: Failed to collect image data.")
        return

    # 3. Perform Data Augmentation and create Generators
    train_generator, val_generator = create_image_data_generators(
        train_df, val_df, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
    )
    
    # ------------------------------------------------------------
    # --- Verification of Image Counts and Generation Status ---
    # ------------------------------------------------------------
    print("\n--- Verification of Data Generators ---")
    
    # A. Check where the images are (i.e., display the first few paths)
    # The source image paths are in the DataFrames.
    print("\n[A] Location of Source Images (first 2 paths from train_df):")
    for path in train_df['image_path'].head(2).tolist():
        print(f"  - {path}")
    print(f"Total Source Images on Disk: {len(train_df) + len(val_df)}\n")

    # B. Check how many images are generated (samples/batches)
    print(f"[B] Generated Samples (Source Count) per Generator:")
    print(f"  - Training Samples (train_generator.samples): {train_generator.samples}")
    print(f"  - Validation Samples (val_generator.samples): {val_generator.samples}")
    print(f"  - Batch Size: {train_generator.batch_size}")

    # Note: Augmentation happens 'on-the-fly' during training. The number of samples
    # below refers to the number of unique images that will be augmented each epoch.
    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)
    
    print(f"\n[B] Batches per Epoch (The number of times augmentation runs per epoch):")
    print(f"  - Training Steps per Epoch: {train_steps}")
    print(f"  - Validation Steps per Epoch: {val_steps}")
    print("--- Verification Complete ---")
    # ------------------------------------------------------------
    
    print("\n Data Preparation Complete. Generators are ready for Transfer Learning.")
    return train_generator, val_generator


# ----------------------------------------------------------------------
# --- Model Building and Training Functions ---
# ----------------------------------------------------------------------

def build_model(architecture, input_shape, num_classes, trainable_layers=50):
    """
    Builds a transfer learning model based on the specified architecture.
    Supported architectures: 'ResNet50', 'EfficientNetB0', 'MobileNetV2'
    Only last `trainable_layers` are trainable; the rest are frozen.
    """
    architecture = architecture.lower()
    
    if architecture == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif architecture == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif architecture == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Freeze all but last `trainable_layers`
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Build top layers
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_transfer")
    
    return model


def train_model(train_generator, val_generator, architecture="ResNet50", epochs=10, lr=0.0001, trainable_layers=20):
    """
    Trains a single model (homogeneous or heterogeneous) with W&B logging.
    """
    print(f"\n--- Training {architecture} model ---")
    
    IMG_HEIGHT, IMG_WIDTH = train_generator.target_size
    num_classes = len(train_generator.class_indices)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = build_model(architecture, input_shape, num_classes, trainable_layers=trainable_layers)

    # W&B configuration
    wandb.config.update({
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_generator.batch_size,
        "model_type": architecture,
        "dataset": "AFHQ"
    })

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[
            wandb.integration.keras.WandbMetricsLogger(log_freq="epoch"),
            wandb.integration.keras.WandbModelCheckpoint("model_checkpoint.keras", save_weights_only=False)
        ]
    )

    final_loss, final_acc = model.evaluate(val_generator, steps=val_steps)
    print(f"{architecture} Validation Accuracy: {final_acc:.4f}")
    wandb.log({f"{architecture}_val_accuracy": final_acc})

    return model


def ensemble_models(models, val_generator):
    """
    Evaluates an ensemble of models on the validation set (homogeneous or heterogeneous).
    Uses soft voting (averaging predictions) and logs W&B metrics.
    """
    val_steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
    print("\n--- Evaluating Ensemble ---")

    # Collect predictions
    preds_list = [model.predict(val_generator, steps=val_steps) for model in models]
    ensemble_preds = np.mean(preds_list, axis=0)
    ensemble_labels = val_generator.classes

    # Compute accuracy
    ensemble_acc = np.mean(np.argmax(ensemble_preds, axis=1) == ensemble_labels)
    print(f"Ensemble Validation Accuracy: {ensemble_acc:.4f}")
    wandb.log({"ensemble_val_accuracy": ensemble_acc})

    return ensemble_preds, ensemble_acc


def homogeneous_ensemble(train_generator, val_generator, num_models=3, epochs=10, lr=0.0001, trainable_layers=50):
    """
    Trains multiple homogeneous ResNet50 models and returns the ensemble results.
    """
    models = []
    for i in range(num_models):
        print(f"\n--- Training homogeneous model {i+1}/{num_models} ---")
        model = train_model(train_generator, val_generator, architecture="ResNet50", epochs=epochs, lr=lr, trainable_layers=trainable_layers)
        models.append(model)

    ensemble_preds, ensemble_acc = ensemble_models(models, val_generator)
    return models, ensemble_preds, ensemble_acc


def heterogeneous_ensemble(train_generator, val_generator, architectures, epochs=10, lr=0.0001, trainable_layers=50):
    """
    Trains multiple heterogeneous models and returns the ensemble results.
    `architectures` is a list of strings, e.g., ["ResNet50", "EfficientNetB0", "MobileNetV2"]
    """
    models = []
    for arch in architectures:
        model = train_model(train_generator, val_generator, architecture=arch, epochs=epochs, lr=lr, trainable_layers=trainable_layers)
        models.append(model)

    ensemble_preds, ensemble_acc = ensemble_models(models, val_generator)
    print(f"\nHeterogeneous Ensemble Accuracy: {ensemble_acc:.4f}")
    wandb.log({"hetero_ensemble_val_accuracy": ensemble_acc})

    return models, ensemble_preds, ensemble_acc


# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Running the preparation pipeline
        train_gen, val_gen = main_prep()
    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")