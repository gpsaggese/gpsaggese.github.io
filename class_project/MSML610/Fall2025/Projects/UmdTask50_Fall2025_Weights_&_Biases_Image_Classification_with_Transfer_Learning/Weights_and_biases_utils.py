import math
import os
import random

import IPython.display as disp
import kagglehub  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.applications
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import wandb  # type: ignore
import wandb.integration.keras  # type: ignore
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # pylint: disable=unused-import

# -----------------------------------------------------------------------------
# --- Constants for Data Configuration ---
# -----------------------------------------------------------------------------
DATASET_REF = "andrewmvd/animal-faces"
BASE_SUBDIR = "afhq"
CLASSES = ["cat", "dog", "wild"]
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 256
num_workers = 16

# -----------------------------------------------------------------------------
# --- Data Preparation Functions ---
# -----------------------------------------------------------------------------


def download_dataset(dataset_ref: str = DATASET_REF) -> str:
    """
    Downloads the specified dataset from KaggleHub.

    Returns     The local file path to the downloaded dataset.
    """
    print(f"Downloading dataset: {dataset_ref}...")
    try:
        path = kagglehub.dataset_download(dataset_ref)
        print("Path to dataset files:", path)
        return path
    except Exception as e:
        print(f"An error occurred during dataset download: {e}")
        return ""


def collect_image_dataframes(
    dataset_path: str, base_subdir: str = BASE_SUBDIR, classes: list = CLASSES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and perform the Train/Validation Split by collecting image paths.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df and val_df with 'image_path' and 'label' columns.
    """
    if not dataset_path:
        return pd.DataFrame(), pd.DataFrame()
    train_data = {"image_path": [], "label": []}
    val_data = {"image_path": [], "label": []}
    print(
        "\nCollecting image paths and labels (performing Train/Validation Split)..."
    )
    for class_name in classes:
        train_dir = os.path.join(dataset_path, base_subdir, "train", class_name)
        val_dir = os.path.join(dataset_path, base_subdir, "val", class_name)
        # Collect train data
        if os.path.exists(train_dir):
            for filename in os.listdir(train_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    train_data["image_path"].append(
                        os.path.join(train_dir, filename)
                    )
                    train_data["label"].append(class_name)
        # Collect validation data
        if os.path.exists(val_dir):
            for filename in os.listdir(val_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    val_data["image_path"].append(os.path.join(val_dir, filename))
                    val_data["label"].append(class_name)
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    print(f"\nTrain data collected. Samples: {len(train_df)}")
    disp.display(train_df.head())
    print(f"\nValidation data collected. Samples: {len(val_df)}")
    disp.display(val_df.head())
    return train_df, val_df


def create_image_data_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_height: int = IMG_HEIGHT,
    img_width: int = IMG_WIDTH,
    batch_size: int = BATCH_SIZE,
) -> tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Perform Data Augmentation and create Keras ImageDataGenerators.

    Returns
    -------
    tuple[ImageDataGenerator, ImageDataGenerator]
        train_generator and val_generator
    """
    print("\nSetting up ImageDataGenerators (including Data Augmentation)...")
    # Data Augmentation for Training Set
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalization (mandatory)
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    # Simple Rescaling for Validation Set (No Augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="label",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="label",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    print("Data Augmentation applied. Generators created successfully.")
    return train_generator, val_generator


def main_prep():
    """
    Execute the data preparation pipeline: download, split, and augment.

    Returns
    -------
    tuple[ImageDataGenerator, ImageDataGenerator] | None
        Training and validation generators, or None if failed.
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
    # -------------------------------------------------------------------------
    # --- Verification of Image Counts and Generation Status ---
    # -------------------------------------------------------------------------
    print("\n--- Verification of Data Generators ---")
    # A. Check where the images are (i.e., display the first few paths)
    # The source image paths are in the DataFrames.
    print("\n[A] Location of Source Images (first 2 paths from train_df):")
    for path in train_df["image_path"].head(2).tolist():
        print(f"  - {path}")
    print(f"Total Source Images on Disk: {len(train_df) + len(val_df)}\n")
    # B. Check how many images are generated (samples/batches)
    print(f"[B] Generated Samples (Source Count) per Generator:")
    print(
        f"  - Training Samples (train_generator.samples): {train_generator.samples}"
    )
    print(
        f"  - Validation Samples (val_generator.samples): {val_generator.samples}"
    )
    print(f"  - Batch Size: {train_generator.batch_size}")
    # Note: Augmentation happens 'on-the-fly' during training. The number of samples
    # below refers to the number of unique images that will be augmented each epoch.
    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)
    print(
        f"\n[B] Batches per Epoch (The number of times augmentation runs per epoch):"
    )
    print(f"  - Training Steps per Epoch: {train_steps}")
    print(f"  - Validation Steps per Epoch: {val_steps}")
    print("--- Verification Complete ---")
    # -------------------------------------------------------------------------
    print(
        "\n Data Preparation Complete. Generators are ready for Transfer Learning."
    )
    return train_generator, val_generator


# -----------------------------------------------------------------------------
# --- Model Building and Training Functions ---
# -----------------------------------------------------------------------------


def build_model(architecture, input_shape, num_classes, trainable_layers=50):
    """
    Build a transfer learning model based on the specified architecture.

    Supported architectures: 'ResNet50', 'EfficientNetB0', 'MobileNetV2'.
    Only last `trainable_layers` are trainable; the rest are frozen.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    architecture = architecture.lower()
    if architecture == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif architecture == "EfficientNetB0":
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
    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name=f"{architecture}_transfer"
    )
    return model


def train_model(
    train_generator,
    val_generator,
    architecture="ResNet50",
    epochs=10,
    lr=0.0001,
    trainable_layers=50,
    unique_name=None,
):
    """
    Train a single model (homogeneous or heterogeneous) with W&B logging.

    Returns
    -------
    tuple[tf.keras.Model, float]
        Trained model and its validation accuracy.
    """
    print(f"\n--- Training {architecture} model ---")
    IMG_HEIGHT, IMG_WIDTH = train_generator.target_size
    num_classes = len(train_generator.class_indices)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = build_model(
        architecture, input_shape, num_classes, trainable_layers=trainable_layers
    )
    # W&B configuration
    run = wandb.init(
        project="animal-faces-classification",
        entity="pshashid-university-of-maryland",
        name=unique_name if unique_name else architecture,
        reinit=True,
    )
    wandb.config.update(
        {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": train_generator.batch_size,
            "model_type": unique_name if unique_name else architecture,
            "dataset": "AFHQ",
        }
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
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
            wandb.integration.keras.WandbModelCheckpoint(
                "model_checkpoint.keras", save_weights_only=False
            ),
        ],
    )
    final_loss, final_acc = model.evaluate(val_generator, steps=val_steps)
    # Use unique_name for distinct logging if available, otherwise use architecture name
    log_key = unique_name if unique_name else architecture
    print(f"{log_key} Validation Accuracy: {final_acc:.4f}")
    # Logging with a unique key
    wandb.log({f"{log_key}_val_accuracy": final_acc})
    run.finish()
    return model, final_acc


def ensemble_models(models, val_generator, ensemble_key="ensemble"):
    """
    Evaluate an ensemble of models on the validation set using soft voting.

    Returns
    -------
    tuple[np.ndarray, float]
        Ensemble predictions and accuracy.
    """
    run = wandb.init(
        project="animal-faces-classification",
        entity="pshashid-university-of-maryland",
        name=ensemble_key,
        reinit=True,
    )
    val_steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
    print(f"\n--- Evaluating {ensemble_key.title()} Ensemble ---")
    # Collect predictions
    preds_list = [
        model.predict(val_generator, steps=val_steps) for model in models
    ]
    ensemble_preds = np.mean(preds_list, axis=0)
    ensemble_labels = val_generator.classes
    # Compute accuracy
    ensemble_acc = np.mean(np.argmax(ensemble_preds, axis=1) == ensemble_labels)
    print(f"{ensemble_key.title()} Validation Accuracy: {ensemble_acc:.4f}")
    # Logging with a unique key
    wandb.log({f"{ensemble_key}_val_accuracy": ensemble_acc})
    run.finish()
    return ensemble_preds, ensemble_acc


def homogeneous_ensemble(
    train_generator,
    val_generator,
    num_models=3,
    epochs=10,
    lr=0.0001,
    trainable_layers=50,
):
    """
    Trains multiple homogeneous ResNet50 models and returns the ensemble
    results.
    """
    models = []
    for i in range(num_models):
        # Create a unique name for each ResNet model instance
        arch_name = f"ResNet50_Homo_{i+1}"
        print(
            f"\n--- Training homogeneous model {i+1}/{num_models} ({arch_name}) ---"
        )
        # Pass the unique name to train_model for distinct logging
        model, _ = train_model(
            train_generator,
            val_generator,
            architecture="ResNet50",
            epochs=epochs,
            lr=lr,
            trainable_layers=trainable_layers,
            unique_name=arch_name,
        )
        models.append(model)
    # Pass unique ensemble key
    ensemble_preds, ensemble_acc = ensemble_models(
        models, val_generator, ensemble_key="homogeneous_ensemble"
    )
    return models, ensemble_preds, ensemble_acc


def heterogeneous_ensemble(
    train_generator,
    val_generator,
    architectures,
    epochs=10,
    lr=0.0001,
    trainable_layers=50,
):
    """
    Train multiple heterogeneous models and return the ensemble results.

    Parameters
    ----------
    architectures : list[str]
        List of architecture names, e.g., ["ResNet50", "EfficientNetB0", "MobileNetV2"].

    Returns
    -------
    tuple[list[tf.keras.Model], np.ndarray, float]
        Trained models, ensemble predictions, and ensemble accuracy.
    """
    models = []
    for arch in architectures:
        # Create a unique name for each model instance, distinguishing it from single/homogeneous runs
        unique_arch_name = f"{arch}_Hetero"
        # Pass the unique name to train_model for distinct logging
        model, _ = train_model(
            train_generator,
            val_generator,
            architecture=arch,
            epochs=epochs,
            lr=lr,
            trainable_layers=trainable_layers,
            unique_name=unique_arch_name,
        )
        models.append(model)
    # Pass unique ensemble key
    ensemble_preds, ensemble_acc = ensemble_models(
        models, val_generator, ensemble_key="heterogeneous_ensemble"
    )
    print(f"\nHeterogeneous Ensemble Accuracy: {ensemble_acc:.4f}")
    return models, ensemble_preds, ensemble_acc


def download_wandb_models_only(
    model_names, entity, project, artifact_dir="models"
):
    """
    Download W&B model artifacts locally in separate subdirectories per model.

    Returns
    -------
    dict[str, str]
        Dictionary mapping model name to local path.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_paths = {}
    api = wandb.Api()  # Use API to avoid logging
    for name in model_names:
        artifact_ref = f"{entity}/{project}/{name}:latest"
        print(f"\nFetching latest artifact: {artifact_ref}")
        artifact = api.artifact(artifact_ref, type="model")
        # Create a unique subdirectory per model
        model_subdir = os.path.join(artifact_dir, name)
        os.makedirs(model_subdir, exist_ok=True)
        local_path = artifact.download(root=model_subdir)
        artifact_paths[name] = local_path
        print(f"Downloaded {name} to {local_path}")
    return artifact_paths


def get_sample_image_path(dataset_path, label, random_state=None):
    """
    Returns the path to a random image of a given label.

    Args:
        dataset_path (str): Root path of the dataset.
        label (str): Class label ('cat', 'dog', etc.).
        random_state (int, optional): Seed for reproducible randomness.

    Returns
        str: Full path to the randomly selected image.
    """
    label_dir = os.path.join(dataset_path, "afhq", "train", label)
    if not os.path.exists(label_dir):
        raise FileNotFoundError(
            f"No folder found for label {label} at {label_dir}"
        )
    img_files = [
        f for f in os.listdir(label_dir) if f.lower().endswith((".jpg", ".png"))
    ]
    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in {label_dir}")
    if random_state is not None:
        random.seed(random_state)
    selected_file = random.choice(img_files)
    return os.path.join(label_dir, selected_file)


def load_and_preprocess(img_path):
    """
    Load an image, resize to (IMG_HEIGHT, IMG_WIDTH), normalize pixels [0,1].

    Returns
    -------
    img_array : np.ndarray
        4D array (1, H, W, 3) for model input.
    pil_img : PIL.Image.Image
        Original PIL Image for visualization.
    """
    pil_img = image.load_img(
        img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="rgb"
    )
    img_array = image.img_to_array(pil_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, pil_img


def load_model_safely(model_dir):
    """
    Load a TensorFlow/Keras model from a checkpoint, with error handling.

    Args:
        model_dir (str): Directory containing the 'model_checkpoint.keras' file.

    Returns
        tf.keras.Model or None: Loaded Keras model if successful, None otherwise.
    """
    checkpoint = os.path.join(model_dir, "model_checkpoint.keras")
    if not os.path.exists(checkpoint):
        print(f"[WARN] Missing checkpoint: {checkpoint}")
        return None
    try:
        model = tf.keras.models.load_model(checkpoint)
        return model
    except Exception as e:
        print(f"[ERROR] Failed loading {checkpoint}: {e}")
        return None


def infer_single_model(model, img_array):
    """
    Run inference on a single model for a given image array.

    Args:
        model (tf.keras.Model): Keras model to use for prediction.
        img_array (np.ndarray): Image array with batch dimension (1, H, W, 3).

    Returns
        np.ndarray: Predicted probabilities for each class.
    """
    return model.predict(img_array, verbose=0)[0]


def run_inference_group(models, model_names, images_dict, group_name):
    """
    Run inference on a group of models for multiple sample images and print
    results. Also computes an ensemble prediction by averaging probabilities.

    Args:
        models (list[tf.keras.Model]): List of models to run.
        model_names (list[str]): Corresponding names of the models.
        images_dict (dict): Dictionary mapping class names to tuples (img_array, pil_img).
        group_name (str): Name of the group (used for printing headings).

    Returns
        None
    """
    print("\n" + "=" * 60)
    print(f" Running Inference for {group_name}")
    print("=" * 60)
    for class_name, (img_array, pil_img) in images_dict.items():
        print(f"\n--- {group_name}: Predictions for {class_name.upper()} ---")
        # Show image
        plt.figure(figsize=(3, 3))
        plt.imshow(pil_img)
        plt.axis("off")
        plt.title(f"{class_name.upper()} image")
        plt.show()
        per_model_probs = []
        for model, name in zip(models, model_names):
            probs = infer_single_model(model, img_array)
            pred_label = CLASSES[np.argmax(probs)]
            per_model_probs.append(probs)
            print(f"{name:<30} → {pred_label} (probs={probs})")
        # Ensemble prediction
        ensemble_probs = np.mean(per_model_probs, axis=0)
        ensemble_label = CLASSES[np.argmax(ensemble_probs)]
        print(
            f"\n[ENSEMBLE {group_name}] → {ensemble_label} (avg_probs={ensemble_probs})"
        )


def load_models(names, artifact_paths):
    """
    Load multiple models from artifact paths, ensuring they accept 3-channel
    input.

    Returns
    -------
    models : list[tf.keras.Model]
        Successfully loaded models.
    loaded_names : list[str]
        Names of successfully loaded models.
    """
    models, loaded_names = [], []
    for name in names:
        model_dir = artifact_paths.get(name)
        if model_dir is None:
            print(f"[WARN] No artifact found for {name}")
            continue
        model = load_model_safely(model_dir)
        if model is None:
            continue
        # Ensure model expects 3 channels
        if model.input_shape[-1] != 3:
            print(f"[WARN] {name} does not expect 3 channels. Skipping.")
            continue
        models.append(model)
        loaded_names.append(name)
    return models, loaded_names


# -----------------------------------------------------------------------------
# --- Main Execution ---
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Running the preparation pipeline
        train_gen, val_gen = main_prep()
    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")
