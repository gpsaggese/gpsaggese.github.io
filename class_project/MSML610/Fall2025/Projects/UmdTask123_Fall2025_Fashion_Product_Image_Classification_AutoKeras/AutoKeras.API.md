# AutoKeras API Notes

These are quick notes on how I'm using the helper functions from
`utils_data_io.py` and `utils_model.py`.

## Loading train / val / test datasets

```python
from utils_data_io import tsv_to_tfds

class_names = [
    "Accessories",
    "Apparel",
    "Footwear",
    "Free Items",
    "Personal Care",
    "Sporting Goods",
]

num_classes = len(class_names)

train_ds = tsv_to_tfds("lists/train.tsv", num_classes)
val_ds   = tsv_to_tfds("lists/val.tsv", num_classes)
test_ds  = tsv_to_tfds("lists/test.tsv", num_classes)

# AutoKeras training

from utils_model import AKImageClassifierAPI

api = AKImageClassifierAPI(max_trials=2, project_name="ak_search")

best_model = api.fit(train_ds, val_ds, epochs=2)

test_loss, test_acc = api.evaluate(test_ds)
print("Accuracy:", test_acc)

api.save("models/autokeras_best.h5", class_names)

# This is the same workflow used inside my notebook.
```
