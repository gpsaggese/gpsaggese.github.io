import os
import traceback
from datasets import load_dataset, DownloadConfig
import sys

DATASET_NAME = "allenai/cord19"
CONFIG_NAME = "metadata"
LOCAL_SAVE_PATH = "/home/haochen/documents/610/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask66_Fall2025_NLTK_Named_Entity_Recognition_in_Scientific_Publications/data"

def download_and_save_dataset():
    print(f"Attempting to download dataset: {DATASET_NAME}...")
    if os.path.exists(LOCAL_SAVE_PATH):
        print(f"Directory '{LOCAL_SAVE_PATH}' already exists. Skipping download.")
        return

    os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)

    # Increase timeout settings to handle large files
    os.environ["HF_DATASETS_TIMEOUT"] = "600"
    os.environ["FSSPEC_HTTP_TIMEOUT"] = "600"

    dl_cfg = DownloadConfig(max_retries=10, use_etag=True, resume_download=True, num_proc=1)

    try:
        print("Start full download with retries...")
        ds = load_dataset(DATASET_NAME, CONFIG_NAME, download_config=dl_cfg)
        print(f"Saving dataset to {LOCAL_SAVE_PATH}...")
        ds.save_to_disk(LOCAL_SAVE_PATH)
        print("✅ Dataset downloaded and saved successfully!")
        return
    except Exception as e:
        print("\nFull download failed, see error details below:")
        print("Error trace:", file=sys.stderr)
        traceback.print_exc()
        print("\n"+ "!" * 50 + "\n")

if __name__ == "__main__":
    download_and_save_dataset()