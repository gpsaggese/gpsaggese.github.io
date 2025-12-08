#!/usr/bin/env python3
"""
RoBERTa Training on Vertex AI (with FULL METRICS)

This script trains Twitter-RoBERTa on Vertex AI for comparison with BERT baseline.

COST: ~$10-15 USD (15-20 minutes on T4 GPU)

Usage:
    python run_roberta_training.py
"""

import vertex_ai_utils as vai

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ID = "noted-cortex-477800-b7"
BUCKET_NAME = "vertex-ai-sentiment-data-msml610"
LOCATION = "us-central1"

# GCS URIs (already uploaded by previous training)
train_gcs = f"gs://{BUCKET_NAME}/sentiment-data/train.jsonl"
val_gcs = f"gs://{BUCKET_NAME}/sentiment-data/val.jsonl"
test_gcs = f"gs://{BUCKET_NAME}/sentiment-data/test.jsonl"


def main():
    print("=" * 60)
    print("ROBERTA TRAINING FOR MODEL COMPARISON")
    print("=" * 60)
    print()
    print("This will train Twitter-RoBERTa on Vertex AI for comparison with BERT.")
    print()
    print("💰 Estimated cost: $10-15 USD")
    print("⏰ Estimated time: 15-20 minutes")
    print()
    
    # Step 1: Initialize Vertex AI
    print("Step 1: Initializing Vertex AI...")
    vai.initialize_vertex_ai(
        project_id=PROJECT_ID,
        location=LOCATION,
        credentials_path="vertex-ai-key.json",
        staging_bucket=f"gs://{BUCKET_NAME}/staging"
    )
    print()
    
    # Step 2: Create RoBERTa training job
    print("Step 2: Creating RoBERTa training job...")
    roberta_job = vai.create_custom_roberta_training_job(
        display_name="sentiment-roberta-fullmetrics",
        script_path="vertex_ai_training.py",
        train_data_gcs_uri=train_gcs,
        val_data_gcs_uri=val_gcs,
        test_data_gcs_uri=test_gcs,
        project_id=PROJECT_ID,
        location=LOCATION,
        learning_rate=2e-5,
        batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_epochs=4
    )
    print()
    
    # Step 3: Run the job
    print("Step 3: Starting RoBERTa training on Vertex AI...")
    print("This will take approximately 15-20 minutes.")
    print("Monitor progress at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    print()
    
    vai.run_roberta_training_job(
        job=roberta_job,
        train_data_gcs_uri=train_gcs,
        val_data_gcs_uri=val_gcs,
        test_data_gcs_uri=test_gcs,
        model_display_name="sentiment-roberta-fullmetrics",
        sync=True  # Wait for completion
    )
    
    print()
    print("=" * 60)
    print("ROBERTA TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print("Review the training logs above for the metrics:")
    print("  - F1-Macro")
    print("  - F1-Weighted")
    print("  - Accuracy")
    print("  - Precision (macro)")
    print("  - Recall (macro)")
    print()
    print("Compare with BERT results for the transfer learning bonus!")


if __name__ == "__main__":
    main()
