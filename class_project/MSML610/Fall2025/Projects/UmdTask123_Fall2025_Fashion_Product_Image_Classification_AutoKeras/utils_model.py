import os
from typing import List, Optional
import autokeras as ak
import tensorflow as tf


class AKImageClassifierAPI:
    """Wrapper for AutoKeras ImageClassifier."""

    def __init__(
        self,
        max_trials: int = 2,
        project_name: str = "ak_search",
        directory: str = "ak_search",
        overwrite: bool = True,
    ) -> None:

        self._ak_model: ak.ImageClassifier = ak.ImageClassifier(
            max_trials=max_trials,
            project_name=project_name,
            directory=directory,
            overwrite=overwrite,
        )

        self._exported: Optional[tf.keras.Model] = None

    def fit(self, train_ds, val_ds, epochs: int = 2):
        """Runs AutoKeras search + trains the best model."""
        self._ak_model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        self._exported = self._ak_model.export_model()
        return self._exported

    def _ensure_exported(self):
        if self._exported is None:
            self._exported = self._ak_model.export_model()
        return self._exported

    def evaluate(self, test_ds):
        model = self._ensure_exported()
        return model.evaluate(test_ds, verbose=0)

    def predict(self, ds):
        model = self._ensure_exported()
        return model.predict(ds)

    def save(self, model_path: str, class_names: List[str]):
        model = self._ensure_exported()

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

        class_file = os.path.join(os.path.dirname(model_path), "class_names.txt")
        with open(class_file, "w", encoding="utf-8") as f:
            for name in class_names:
                f.write(name + "\n")
