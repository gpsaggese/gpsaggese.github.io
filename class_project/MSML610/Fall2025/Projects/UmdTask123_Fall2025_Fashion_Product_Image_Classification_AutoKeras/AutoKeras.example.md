````markdown
# AutoKeras Example – Full Workflow Notes

This file summarizes what I implemented inside `AutoKeras.example.ipynb`.

## Step 1 – Imports

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from utils_data_io import tsv_to_tfds
from utils_model import AKImageClassifierAPI

## Step 2 – Inspect styles.csv

df = pd.read_csv("styles.csv", on_bad_lines="skip")
print(df.head())

## Step 3 – Filter categories and build labels

target_categories = [
    "Accessories",
    "Apparel",
    "Footwear",
    "Free Items",
    "Personal Care",
    "Sporting Goods",
]

df = df[df.masterCategory.isin(target_categories)].copy()
df["image_path"] = "images/" + df["id"].astype(str) + ".jpg"
df["label_name"] = df["masterCategory"]

class_names = sorted(df["label_name"].unique())
class_to_idx = {c: i for i, c in enumerate(class_names)}
df["label_idx"] = df["label_name"].map(class_to_idx)

## Step 4 – Keep only rows with existing images

df = df[df["image_path"].apply(os.path.exists)].copy()


## Step 5 – Split and save TSV files

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label_idx"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_idx"], random_state=42
)

os.makedirs("lists", exist_ok=True)

for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split[["image_path", "label_idx"]].to_csv(
        f"lists/{name}.tsv",
        sep="\t",
        header=False,
        index=False,
    )


## Step 6 – Create tf.data datasets

num_classes = len(class_names)

train_ds = tsv_to_tfds("lists/train.tsv", num_classes)
val_ds   = tsv_to_tfds("lists/val.tsv", num_classes)
test_ds  = tsv_to_tfds("lists/test.tsv", num_classes)


## Step 7 – Train AutoKeras

api = AKImageClassifierAPI(max_trials=2, project_name="ak_search")
model = api.fit(train_ds, val_ds, epochs=2)

api.save("models/autokeras_best.h5", class_names)

## Step 8 – Evaluation and reports

preds = []
true = []

for x, y in test_ds:
    p = model.predict(x)
    preds.extend(np.argmax(p, axis=1))
    true.extend(y.numpy())

report = classification_report(true, preds, target_names=class_names)

os.makedirs("outputs", exist_ok=True)

with open("outputs/report_autokeras.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(true, preds)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confmat_autokeras.png")
plt.close()
```
````
