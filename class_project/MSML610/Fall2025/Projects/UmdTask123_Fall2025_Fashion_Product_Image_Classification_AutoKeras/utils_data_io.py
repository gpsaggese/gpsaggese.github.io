import os
from typing import Tuple
import pandas as pd
import tensorflow as tf


def load_styles(styles_path: str) -> pd.DataFrame:
    """Simple helper to load styles.csv."""
    return pd.read_csv(styles_path)


def _load_and_preprocess_image(img_path: tf.Tensor,
                               img_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """Reads and preprocesses an image."""
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def tsv_to_tfds(tsv_path: str,
                num_classes: int,
                img_size: Tuple[int, int] = (224, 224),
                batch: int = 32,
                shuffle: bool = True) -> tf.data.Dataset:
    """
    Converts a TSV file with [image_path, label_idx] into a tf.data.Dataset.
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["image_path", "label_idx"]
    )

    img_paths = df["image_path"].astype(str).values
    labels = df["label_idx"].astype("int32").values

    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    def _map_fn(rel_path, label):
        img = _load_and_preprocess_image(rel_path, img_size)
        return img, tf.cast(label, tf.int32)

    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(len(df))

    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return ds
