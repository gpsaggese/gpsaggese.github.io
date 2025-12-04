# AutoKeras API – Fashion Product Image Classification

## Project goal

The goal of this project was to use **AutoKeras** to build an image classification model for fashion products.  
Instead of manually designing a CNN, I let AutoKeras automatically search for the best working architecture using a subset of the **Myntra Fashion Product Images dataset** from Kaggle.

The target task is to classify images into one of these six categories:

- Accessories
- Apparel
- Footwear
- Free Items
- Personal Care
- Sporting Goods

---

## Dataset

I used the dataset **"Fashion Product Images (Small)"** from Kaggle.  
It includes over 44,000 product images along with a metadata CSV file (`styles.csv`) that provides category labels.

After loading the CSV file, I:

- Filtered the rows to include only the six target categories.
- Built full local image paths for each product.
- Mapped string labels to integer class indices (`label_idx`).
- Removed any entries where the referenced image file was missing.

This resulted in approximately **44,418 valid labeled images** before any downsampling.

---

## Why downsampling was necessary

Training directly on the full dataset was not possible inside **Colab** due to memory limits.  
Early attempts caused repeated kernel crashes during dataset loading and conversion.

To solve this:

- I created a **class-balanced subset** of the data.
- Sampled **up to 1,000 images per class** using random sampling.
- This produced a final working dataset of **4,130 images total**.

This approach preserved class balance while keeping the problem small enough to fit into Colab GPU memory.

---

## Data splitting

The balanced dataset was split as follows:

- **80% – Training**
- **10% – Validation**
- **10% – Test**

The split was stratified to maintain category proportions across all datasets.

The final sizes after splitting:

- Training: 3304 images
- Validation: 413 images
- Test: 413 images

---

## Dataset pipeline

Each dataset split was saved as a TSV file (`train.tsv`, `val.tsv`, `test.tsv`) containing: image_path label_index

A helper function from `utils_data_io.py` was used to build a **TensorFlow `tf.data.Dataset` pipeline** that:

- Loads images from disk.
- Resizes them to 224×224.
- Normalizes pixel values.
- Applies batching and prefetching.

---

## Conversion to NumPy

AutoKeras works more consistently with NumPy arrays than with raw TensorFlow datasets, but converting the full dataset into NumPy caused memory crashes.

To avoid this:

- I converted only a limited number of samples:
  - 2,000 images for training
  - 500 images for validation
  - 500 images for testing

This kept memory usage stable while still providing enough data for meaningful training and evaluation.

---

## AutoKeras training setup

I used AutoKeras’ `ImageClassifier` API for model search and training.

### Search parameters:

- **max_trials = 3**  
  The number of candidate CNN architectures AutoKeras evaluates.

- **epochs = 2 per trial**  
  Each architecture was trained briefly to evaluate validation performance.

These values were chosen specifically to **stay within Colab time and memory limits**, while still allowing the system to test multiple architectures and converge on a good model.

---

## Training outcome

AutoKeras completed all 3 trials successfully and selected the model with the lowest validation loss.

Key results:

- Final test accuracy: **~91%**
- Best validation loss: **~0.33**
- Strong performance on major classes including Apparel, Accessories, Footwear, and Personal Care.

Two smaller classes (**Free Items** and **Sporting Goods**) had very few samples even after downsampling (105 and 25 images respectively), which caused lower precision/recall scores in the classification report. This class imbalance explains why macro-averaged metrics are lower than weighted averages.

---

## Model saving

After training, the best architecture was exported and saved to disk as: models/autokeras_best.h5

This file preserves the network weights and architecture so training does not need to be repeated.

---

## Evaluation and outputs

The saved model was reloaded and evaluated against the test dataset.

The following evaluation artifacts were produced and stored:

- **Classification report**  
  `outputs/report_autokeras.txt`

- **Confusion matrix plot**  
  `outputs/confmat_autokeras.png`

These outputs summarize:

- Per-class precision, recall, F1 scores
- Overall accuracy
- Visual distribution of predictions vs actual labels

---

## Summary

In this notebook, I built a complete AutoKeras pipeline including:

- Dataset filtering and cleaning
- Balanced downsampling to manage memory
- Stratified train/val/test splitting
- TensorFlow dataset building
- NumPy conversion for stable training
- Automated model search using AutoKeras
- Test evaluation and result visualization

Despite hardware constraints, the final model reached **~91% accuracy** and demonstrates that AutoKeras can quickly produce strong CNN baselines without manual architecture engineering.
