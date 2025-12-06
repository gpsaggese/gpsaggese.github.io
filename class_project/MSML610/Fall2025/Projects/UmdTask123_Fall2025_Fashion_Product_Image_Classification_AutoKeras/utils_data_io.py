import tensorflow as tf
import pandas as pd
import numpy as np
import os

def tsv_to_tfds(tsv_path, num_classes):
    """
    Convert a TSV file into a tf.data.Dataset where each row is:
      image_path<TAB>label_idx

    Returns a dataset of (image_tensor, label_int).
    """

    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["image_path", "label_idx"])

    paths = df["image_path"].tolist()
    labels = df["label_idx"].astype(int).tolist()

    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        label = tf.cast(label, tf.int32) 
        return img, label

    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def ds_to_numpy(ds, max_samples=1000):
    """
    Convert the first `max_samples` from a tf.data.Dataset
    into NumPy arrays: (X, y)
    """
    X_list, y_list = [], []
    count = 0

    for img, label in ds:
        X_list.append(img.numpy())
        y_list.append(label.numpy())
        count += 1
        if count >= max_samples:
            break

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y
