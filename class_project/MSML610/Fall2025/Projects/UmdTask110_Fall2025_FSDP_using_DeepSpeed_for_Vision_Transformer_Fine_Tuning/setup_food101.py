#!/usr/bin/env python3
"""
Script to download and setup the Food-101 dataset.
Downloads the dataset and extracts it to the specified directory.
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, destination, desc=None):
    """Download a file with progress bar."""
    desc = desc or os.path.basename(url)
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)


def organize_food101_splits(extract_dir, dataset_dir, use_symlinks=True):
    """
    Organize Food-101 dataset into train and test (validation) splits.
    
    Args:
        extract_dir: Directory where the dataset was extracted
        dataset_dir: Base dataset directory
        use_symlinks: If True, create symlinks instead of copying files (saves space)
    """
    extract_dir = Path(extract_dir)
    dataset_dir = Path(dataset_dir)
    
    images_dir = extract_dir / "images"
    meta_dir = extract_dir / "meta"
    
    train_list_file = meta_dir / "train.txt"
    test_list_file = meta_dir / "test.txt"
    
    if not (train_list_file.exists() and test_list_file.exists()):
        print("⚠ Warning: train.txt or test.txt not found in meta directory")
        return
    
    # Create train and test directories
    train_dir = dataset_dir / "food-101" / "train"
    test_dir = dataset_dir / "food-101" / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Read train and test splits
    def read_split_file(split_file):
        """Read split file and return list of (class, image_id) tuples."""
        with open(split_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Format: class_name/image_id
        return [line.split('/') for line in lines]
    
    train_items = read_split_file(train_list_file)
    test_items = read_split_file(test_list_file)
    
    print(f"  - Train images: {len(train_items)}")
    print(f"  - Test images: {len(test_items)}")
    
    # Organize train split
    print("  Organizing train split...")
    for class_name, image_id in tqdm(train_items, desc="  Creating train links", unit="images"):
        src = images_dir / class_name / f"{image_id}.jpg"
        dst_class_dir = train_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_class_dir / f"{image_id}.jpg"
        
        if not dst.exists():
            if use_symlinks:
                try:
                    # Use absolute path for symlink
                    dst.symlink_to(src.resolve())
                except OSError:
                    # Fallback to copy if symlink fails (e.g., on Windows or across filesystems)
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
    
    # Organize test split (use as validation)
    print("  Organizing test split (validation)...")
    for class_name, image_id in tqdm(test_items, desc="  Creating test links", unit="images"):
        src = images_dir / class_name / f"{image_id}.jpg"
        dst_class_dir = test_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_class_dir / f"{image_id}.jpg"
        
        if not dst.exists():
            if use_symlinks:
                try:
                    dst.symlink_to(src)
                except OSError:
                    # Fallback to copy if symlink fails
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
    
    # Count organized images
    train_count = sum(1 for _ in train_dir.rglob("*.jpg"))
    test_count = sum(1 for _ in test_dir.rglob("*.jpg"))
    
    print(f"\n✓ Data organization complete!")
    print(f"  - Train directory: {train_dir}")
    print(f"    - Images: {train_count}")
    print(f"  - Test/Validation directory: {test_dir}")
    print(f"    - Images: {test_count}")


def setup_food101(dataset_dir):
    """
    Download and extract Food-101 dataset.
    
    Args:
        dataset_dir: Directory where the dataset will be stored
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Food-101 dataset URL
    dataset_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    tar_filename = dataset_dir / "food-101.tar.gz"
    
    print(f"Downloading Food-101 dataset to {dataset_dir}...")
    print(f"URL: {dataset_url}")
    
    # Download the dataset
    if not tar_filename.exists():
        print("Downloading dataset (this may take a while, ~4.3 GB)...")
        download_file(dataset_url, tar_filename, desc="Food-101")
    else:
        print(f"Dataset archive already exists at {tar_filename}")
        print("Skipping download. Delete the file if you want to re-download.")
    
    # Extract the dataset
    extract_dir = dataset_dir / "food-101"
    if extract_dir.exists():
        print(f"Dataset already extracted at {extract_dir}")
        print("Skipping extraction. Delete the directory if you want to re-extract.")
    else:
        print(f"Extracting dataset to {extract_dir}...")
        with tarfile.open(tar_filename, 'r:gz') as tar:
            # Show progress for extraction
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting", unit="files"):
                tar.extract(member, dataset_dir)
        
        print(f"Extraction complete!")
    
    # Verify the dataset structure
    images_dir = extract_dir / "images"
    meta_dir = extract_dir / "meta"
    
    if not (images_dir.exists() and meta_dir.exists()):
        print(f"\n⚠ Warning: Expected directories not found. Please check {extract_dir}")
        return extract_dir
    
    print(f"\n✓ Dataset extracted successfully")
    print(f"  - Images directory: {images_dir}")
    print(f"  - Meta directory: {meta_dir}")
    
    # Organize data into train and test (validation) splits
    print("\nOrganizing data into train and test splits...")
    organize_food101_splits(extract_dir, dataset_dir)
    
    return extract_dir


if __name__ == "__main__":
    # Set the target directory
    target_dir = "/home/siddpath/scratch.msml610/datasets"
    
    print("=" * 60)
    print("Food-101 Dataset Setup")
    print("=" * 60)
    
    dataset_path = setup_food101(target_dir)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

