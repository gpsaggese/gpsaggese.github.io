# JAX_wildlife.example
End-to-end workflow for Animals-10: load data, train the CNN, evaluate metrics, and persist diagnostics when running inside Docker.

## Flow
1. **Data prep** – Call `load_dataset()` from `JAX_wildlife_utils.py` with `image_size=(128,128)` and the 70/15/15 split
2. **Training** – Configure `TrainConfig(num_epochs=3, batch_size=64)` to keep runtime manageable on CPU. 
3. **Evaluation** – Use `evaluate()` for accuracy/precision/recall + confusion matrix, then `plot_confusion_matrix()` to visualize


