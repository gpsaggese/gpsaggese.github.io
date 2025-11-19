# UmdTask123 – Fashion Product Image Classification with AutoKeras

This project uses the **Fashion Product Images (Small)** dataset from Kaggle to build an image classifier for fashion items using **AutoKeras**.

## Dataset

- Source: Kaggle – “Fashion Product Images (Small)”
- Images are stored as `images/{id}.jpg`
- Metadata file: `styles.csv`

I filtered the dataset to only the following `masterCategory` classes:

- Accessories
- Apparel
- Footwear
- Free Items
- Personal Care
- Sporting Goods

Missing images were removed, and the dataset was split into train/val/test using stratified sampling.  
The splits are stored in:

lists/train.tsv
lists/val.tsv
lists/test.tsv

Each TSV contains:

image_path label_idx

## Project Progress

So far I have:

1. Cleaned and filtered `styles.csv`
2. Verified and linked image IDs to filenames
3. Created deterministic train/val/test splits
4. Built data loaders using `utils_data_io.py`
5. Trained an AutoKeras classifier:
   - max_trials = 2
   - epochs = 2
6. Exported the best model to:
   - `models/autokeras_best.h5`
7. Generated evaluation outputs:
   - `outputs/report_autokeras.txt`
   - `outputs/confmat_autokeras.png`

## Main Files

- `AutoKeras.example.ipynb` → End-to-end training and evaluation
- `AutoKeras.API.md` → Notes on helper API usage
- `AutoKeras.example.md` → Text version of the notebook flow
- `utils_data_io.py` → Loads TSV → tf.data.Dataset
- `utils_model.py` → Wrapper around AutoKeras ImageClassifier
