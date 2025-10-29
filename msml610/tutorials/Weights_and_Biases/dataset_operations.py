import kagglehub
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import IPython.display as disp 
import PIL.Image

# --- Constants for Data Configuration ---
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


# ----------------------------------------------------------------------
# --- Main Execution for Data Preparation ---
# ----------------------------------------------------------------------

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
    import math
    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)
    
    print(f"\n[B] Batches per Epoch (The number of times augmentation runs per epoch):")
    print(f"  - Training Steps per Epoch: {train_steps}")
    print(f"  - Validation Steps per Epoch: {val_steps}")
    print("--- Verification Complete ---")
    # ------------------------------------------------------------
    
    print("\n Data Preparation Complete. Generators are ready for Transfer Learning.")
    return train_generator, val_generator

if __name__ == "__main__":
    try:
        # Running the preparation pipeline
        train_gen, val_gen = main_prep()
    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")