# **PEFT Sentiment Analysis — API Documentation**

This document explains the core technologies and APIs used for **Parameter-Efficient Fine-Tuning (PEFT)** of transformer models for sentiment analysis and text classification tasks.

This is a tool-focused guide that explains:

- What PEFT and LoRA are
- How to use HuggingFace Transformers APIs
- How to apply PEFT/LoRA to any text classification task
- Key APIs and their usage patterns

For a complete project implementation example, see [PEFT_Sentiment_Analysis.example.md](PEFT_Sentiment_Analysis.example.md).

---

# 🔍 **1. What is PEFT?**

**PEFT (Parameter-Efficient Fine-Tuning)** enables training large language models by updating only a small fraction of parameters, rather than fine-tuning the entire model.

**Why use PEFT?**

- Reduces trainable parameters by 90-99%
- Significantly lower memory requirements
- Faster training times
- Maintains model performance
- Enables fine-tuning on consumer hardware

---

# 🛠️ **2. Technologies Covered**

# 🛠️ **2. Technologies Covered**

This tutorial introduces three powerful components for text classification:

---

## **2.1 HuggingFace Transformers**

A library providing state-of-the-art NLP models with a unified API.

**Key Components:**

- **`RobertaTokenizer`** — Converts text to token IDs using Byte-Pair Encoding (BPE)
- **`RobertaForSequenceClassification`** — Pre-trained encoder with classification head
- **`Dataset`** — Lightweight, memory-efficient data handling
- **`Trainer`** & **`TrainingArguments`** — High-level training API

**Installation:**

```bash
pip install transformers
```

---

## **2.2 PEFT (Parameter-Efficient Fine-Tuning)**

PEFT enables training large models without updating all weights.

**LoRA (Low-Rank Adapters)** - The specific PEFT method used:

- Injects small trainable matrices into attention layers
- Reduces trainable parameters from ~125M → ~800K (99.4% reduction)
- Fast, low-cost, GPU/CPU friendly

**Installation:**

```bash
pip install peft
```

---

## **2.3 HuggingFace Datasets**

Provides clean, memory-efficient datasets for PyTorch models with automatic batching and preprocessing.

**Installation:**

```bash
pip install datasets
```

---

# 📦 **3. What Problems Does This Solve?**

# 📦 **3. What Problems Does This Solve?**

Traditional transformer fine-tuning requires:

- Large compute resources (powerful GPUs)
- High memory usage (storing gradients for all parameters)
- Long training times
- Complex API management

**PEFT with LoRA solves these problems by:**

- Training only 0.7% of parameters
- Reducing memory footprint by 90%+
- Enabling CPU-based fine-tuning
- Maintaining competitive model performance
- Simplifying deployment (small adapter files vs. full model weights)

---

# 🔧 **4. Core APIs Used**

# 🔧 **4. Core APIs Used**

---

## **4.1 Tokenization API**

**Purpose:** Convert raw text into numerical token IDs that models can process.

```python
from transformers import RobertaTokenizer

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize text
encoded = tokenizer(
    "This movie was excellent!",
    padding="max_length",
    truncation=True,
    max_length=128
)
```

**Key Parameters:**

- `padding`: Pad sequences to same length ("max_length", "longest", or False)
- `truncation`: Cut off sequences longer than max_length
- `max_length`: Maximum sequence length

**Returns:**

- `input_ids`: Token IDs
- `attention_mask`: Mask indicating real tokens vs. padding

---

## **4.2 Dataset API**

**Purpose:** Efficient data handling with built-in batching and preprocessing.

```python
from datasets import Dataset

# Create dataset from dictionary
dataset = Dataset.from_dict({
    "text": ["Sample text 1", "Sample text 2"],
    "label": [0, 1]
})

# Apply preprocessing
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
```

**Key Methods:**

- `from_dict()`: Create dataset from Python dictionary
- `map()`: Apply function to all examples (supports batching)
- `train_test_split()`: Split into train/test sets

---

## **4.3 Model API**

**Purpose:** Load pre-trained models for text classification.

```python
from transformers import RobertaForSequenceClassification

# Load model
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2  # Binary classification
)
```

**Key Parameters:**

- `model_name`: Pre-trained model identifier (e.g., "roberta-base", "bert-base-uncased")
- `num_labels`: Number of classification categories

---

## **4.4 PEFT / LoRA API**

**Purpose:** Apply parameter-efficient fine-tuning adapters to models.

```python
from peft import LoraConfig, TaskType, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,      # Sequence classification
    r=8,                              # Rank (adapter dimension)
    lora_alpha=16,                    # Scaling factor
    lora_dropout=0.1,                 # Dropout for regularization
    bias="none",                      # Don't train bias terms
    target_modules=["query", "value"] # Which layers to adapt
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # Shows parameter reduction
```

**Key Parameters:**

- `r`: Rank of adaptation matrices (typical: 4, 8, 16)
- `lora_alpha`: Scaling factor (typically 2\*r)
- `target_modules`: Which attention layers to modify (["query", "value"] is common)

---

## **4.5 Training API**

**Purpose:** Simplify the training loop with automatic batching and optimization.

```python
from transformers import TrainingArguments, Trainer

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
)

# Create trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()
```

**Key Parameters:**

- `num_train_epochs`: Number of complete passes through dataset
- `per_device_train_batch_size`: Samples per batch (adjust for memory)
- `learning_rate`: Optimizer learning rate (typical: 1e-5 to 5e-5)
- `evaluation_strategy`: When to evaluate ("epoch", "steps", or "no")

---

# 📊 **5. Complete Workflow Example**

Here's how all the APIs work together:

```python
# 1. Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 2. Prepare dataset
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# 3. Apply LoRA
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)

# 4. Train
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 5. Evaluate
predictions = trainer.predict(test_dataset)
```

---

# 🎯 **6. Use Cases**

This API approach is ideal for:

- **Sentiment Analysis**: Movie reviews, product reviews, social media
- **Text Classification**: News categorization, spam detection, topic labeling
- **Binary Classification**: Fake news detection, toxicity detection
- **Multi-class Classification**: Emotion detection, intent classification

---

# 🧠 **7. LoRA vs. Other Approaches**

| Approach               | Trainable Params  | Memory Usage | Training Time | Accuracy |
| ---------------------- | ----------------- | ------------ | ------------- | -------- |
| **Full Fine-Tuning**   | 100% (~125M)      | Very High    | Slow          | Highest  |
| **Feature Extraction** | ~0.1% (head only) | Low          | Fast          | Lower    |
| **Adapter Layers**     | ~3-5%             | Medium       | Medium        | High     |
| **Prompt Tuning**      | ~0.01%            | Very Low     | Very Fast     | Medium   |
| **LoRA** ⭐            | ~0.7%             | Low          | Fast          | High     |

**LoRA provides the best balance** of efficiency, performance, and ease of use.

---

# 📚 **8. References & Resources**

**Documentation:**

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)

**Papers:**

- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- RoBERTa: [A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

**Tutorials:**

- [PEFT Examples](https://github.com/huggingface/peft/tree/main/examples)
- [Fine-tuning with Trainer](https://huggingface.co/docs/transformers/training)

---

# 🚀 **9. Getting Started**

**Installation:**

```bash
pip install transformers peft datasets torch
```

**Quick Start:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 1. Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 2. Apply LoRA
config = LoraConfig(task_type="SEQ_CLS", r=8)
model = get_peft_model(model, config)

# 3. Prepare your data and train (see example notebook)
```

For a complete end-to-end implementation, see [PEFT_Sentiment_Analysis.example.md](PEFT_Sentiment_Analysis.example.md).

---
