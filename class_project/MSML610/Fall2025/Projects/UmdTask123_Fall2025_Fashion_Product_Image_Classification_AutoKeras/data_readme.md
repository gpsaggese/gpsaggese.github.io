# Data Notes

Dataset: Kaggle “Fashion Product Images (Small)”.

Images are stored locally as:

images/{id}.jpg

The project does NOT commit actual images to Git.

Train/Val/Test splits are stored in `lists/`:

- `train.tsv`
- `val.tsv`
- `test.tsv`

Each TSV has 2 tab-separated columns:

relative_image_path label_idx

Example:

images/12345.jpg 2

These TSV files are consumed by `tsv_to_tfds()` inside `utils_data_io.py`.
