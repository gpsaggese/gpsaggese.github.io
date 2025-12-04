# AutoKeras Example Notebook – Baseline CNN Comparison

## Purpose of this Notebook

This notebook builds a **manual baseline CNN model** and compares its performance against the AutoKeras model trained in `AutoKeras.API.ipynb`.  
The goal is to show that AutoKeras automatically discovered architecture can outperform a **simple hand-designed CNN**, and to evaluate both approaches using the same test data.

---

## Dataset

I used the **Fashion Product Images (Small)** dataset from Kaggle.  
Only six high-level classes were used for classification:

- Accessories
- Apparel
- Footwear
- Free Items
- Personal Care
- Sporting Goods

To avoid file-loading crashes in Colab:

- TSV split files (`lists/train.tsv`, `val.tsv`, `test.tsv`) were cleaned to remove any accidental headers.
- Rows pointing to missing image files were removed.
- For memory stability, datasets were **converted from `tf.data.Dataset` to NumPy arrays** with capped sample sizes:
  - ~2000 training images
  - ~500 validation images
  - ~500 testing images

This kept GPU memory usage within limits while preserving class diversity.

---

## Baseline CNN Design

The baseline model was intentionally kept **simple and interpretable**:

**Architecture**

- Input rescaling (1/255 normalization)
- 3 convolution blocks:
  - Conv2D → ReLU → MaxPooling
  - Filters: 32 → 64 → 128
- Flatten layer
- Dense(256) + ReLU
- Dropout(0.5)
- Dense(6) Softmax output

**Training Setup**

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 5

The goal was not to over-engineer the baseline but to create a **reasonable reference point** for comparison with AutoKeras performance.

---

## Evaluation

The baseline CNN was evaluated using:

- Accuracy
- Per-class Precision
- Recall
- F1-score
- Confusion Matrix

Outputs generated and saved:

- `outputs/report_baseline.txt`
- `outputs/confmat_baseline.png`

Some classes had very small sample sizes (e.g., _Free Items_, _Sporting Goods_), which resulted in unstable precision/recall for those categories.  
This imbalance highlights one limitation of hand-crafted models on smaller subsets and motivates the use of AutoML approaches.

---

## Comparison with AutoKeras

The AutoKeras model (from `AutoKeras.API.ipynb`) achieved:

- **Higher overall validation and test accuracy**
- Better performance on under-represented classes
- Automatically selected deeper CNN patterns than the manual baseline

Despite using fewer epochs and more constrained memory limits, the AutoKeras architecture consistently matched or exceeded baseline performance.

---

## Key Takeaways

- A simple CNN baseline is useful for establishing a performance reference.
- Manual tuning requires careful design choices and still struggles with class imbalance.
- **AutoKeras provided a stronger model with minimal architecture engineering**, making it better suited for this classification task under limited development time.

Overall, the comparison demonstrates why automated neural architecture search is effective for real-world image classification problems.

---

## Colab Constraints

All experiments were bounded by **Colab memory limits**.  
For both models, training sizes, epochs, and trial counts were intentionally capped to maintain stability and prevent GPU/CPU crashes.
