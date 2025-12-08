#!/usr/bin/env python3
"""
BERT Baseline Training on Vertex AI

BONUS REQUIREMENT: "Explore transfer learning with pre-trained models like BERT"

This script trains bert-base-uncased on Vertex AI for comparison with Twitter-RoBERTa.
Run this to complete the bonus requirement.

COST: ~$10-15 USD (15-20 minutes on T4 GPU)

Usage:
    python run_bert_training.py
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
    print("BERT BASELINE TRAINING FOR BONUS REQUIREMENT")
    print("=" * 60)
    print()
    print("This will train bert-base-uncased on Vertex AI for comparison with RoBERTa.")
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
    
    # Step 2: Create BERT training job
    print("Step 2: Creating BERT training job...")
    bert_job = vai.create_custom_bert_training_job(
        display_name="sentiment-bert-baseline",
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
    print("Step 3: Starting BERT training on Vertex AI...")
    print("This will take approximately 15-20 minutes.")
    print("Monitor progress at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    print()
    
    vai.run_bert_training_job(
        job=bert_job,
        train_data_gcs_uri=train_gcs,
        val_data_gcs_uri=val_gcs,
        test_data_gcs_uri=test_gcs,
        model_display_name="sentiment-bert-baseline",
        sync=True  # Wait for completion
    )
    
    print()
    print("=" * 60)
    print("BERT TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print("Review the training logs above for the F1-score and confusion matrix.")
    print("Compare with RoBERTa results to complete the bonus requirement:")
    print()
    print("Expected Results:")
    print("  BERT F1-Macro:    ~0.75-0.78")
    print("  RoBERTa F1-Macro: ~0.78-0.85")
    print()
    print("RoBERTa should outperform BERT because it was pre-trained on 124M tweets,")
    print("making it better suited for social media sentiment analysis.")


if __name__ == "__main__":
    main()
