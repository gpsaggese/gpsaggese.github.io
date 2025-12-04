import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import LoraConfig, get_peft_model, TaskType

import shap
from transformers import pipeline


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_fake_true(fake_path="Data/Fake.csv", true_path="Data/True.csv"):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    return df


# ============================================================
# 2. TEXT CLEANING AND PREPROCESSING
# ============================================================

def remove_punct(text):
    return "".join([char for char in text if char not in string.punctuation])

def preprocess_text(df):
    df["text_clean"] = df["text"].apply(lambda x: remove_punct(x.lower()))

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    stopwords_en = stopwords.words("english")
    df["text_tokens"] = df["text_clean"].apply(word_tokenize)

    df["text_tokens"] = df["text_tokens"].apply(
        lambda tokens: [w for w in tokens if w not in stopwords_en]
    )

    wn = WordNetLemmatizer()
    df["text_tokens"] = df["text_tokens"].apply(
        lambda tokens: [wn.lemmatize(w) for w in tokens]
    )

    df["text_final"] = df["text_tokens"].apply(lambda tokens: " ".join(tokens))

    return df


# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================

def split_data(df):
    X = df["text_final"].tolist()
    y = df["label"].tolist()

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# ============================================================
# 4. TOKENIZATION + HUGGINGFACE DATASETS
# ============================================================

def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

def prepare_hf_dataset(train_texts, test_texts, train_labels, test_labels):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    train_dataset = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        batch_size=len(train_dataset)
    )
    test_dataset = test_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        batch_size=len(test_dataset)
    )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, test_dataset, tokenizer


# ============================================================
# 5. LOAD MODEL + APPLY LORA
# ============================================================

def load_roberta_lora():
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================
# 6. TRAINING SETUP
# ============================================================

def get_training_args():
    return TrainingArguments(
        output_dir="./roberta_lora_results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        max_steps=800,
        save_steps=200,
        logging_dir="./logs",
        logging_steps=50
    )

def get_trainer(model, train_dataset, test_dataset, training_args):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds)}

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )


# ============================================================
# 7. EVALUATION HELPERS
# ============================================================

def evaluate_model(trainer, test_dataset, test_labels):
    preds_output = trainer.predict(test_dataset)
    y_pred = preds_output.predictions.argmax(-1)
    y_true = test_labels
    y_prob = preds_output.predictions[:, 1]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# ============================================================
# 8. SHAP EXPLAINABILITY
# ============================================================

def setup_shap(model, tokenizer):
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True,
        max_length=256
    )
    return shap.Explainer(classifier)

def shap_explain(explainer, text):
    shap_values = explainer([text])
    return shap_values
