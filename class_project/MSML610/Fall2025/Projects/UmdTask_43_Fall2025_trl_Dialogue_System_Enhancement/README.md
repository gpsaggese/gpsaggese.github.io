## Name: Venkata Siva Rajesh Vithanala
## UID: 121330543
## Diffuclty level: 2 


# Dialogue System Enhancement 

This project fine-tunes a pre-trained conversational model using **reinforcement learning with human feedback (RLHF)** to enhance dialogue response quality. The optimization uses **Proximal Policy Optimization (PPO)** from the `trl` library and targets **user satisfaction** as defined by sentiment and fluency metrics.

---

## Dataset

We use the **`pixelsandpointers/daily_dialog_w_turn_templates`** dataset from Hugging Face to train and evaluate our dialogue system enhancement pipeline.

This dataset is derived from DailyDialog-style conversations and is preprocessed into **adjacent turn pairs**, making it well-suited for conversational response generation tasks.

**Key characteristics:**
- Total size: ~90,000 utterance pairs
- Training split: ~76,000 examples
- Format: Parquet
- Modality: Text

**Schema:**
- `first` (string): preceding conversational utterance (context)
- `second` (string): corresponding reply utterance (target response)
- `labels` (int): turn-template or dialogue relation label (not used for generation)

Unlike multi-turn dialogue datasets that require explicit context window construction, each row in this dataset already represents a single **context–response pair**. This simplifies preprocessing and allows the model to focus directly on learning coherent and relevant replies.

In our system:
- `first` is used as the input prompt
- `second` is used as the ground-truth response

The `labels` field is ignored during training, as our objective is response quality enhancement rather than dialogue act classification.

This dataset provides a strong foundation for supervised fine-tuning and reinforcement learning–based dialogue system improvement.

---


## Model

The dialogue system is built using **`microsoft/DialoGPT-small`**, a transformer-based conversational language model released by Microsoft and hosted on Hugging Face.

DialoGPT is based on the GPT-2 architecture and is trained using a causal language modeling objective, where the model learns to predict the next token given all previous tokens in a conversation. This design makes it particularly effective for open-domain dialogue generation, as it can maintain conversational flow and generate contextually relevant responses.

The model is pre-trained on a large corpus of multi-turn conversations extracted from online discussion forums, enabling it to capture common conversational patterns such as question–answer exchanges, clarifications, and follow-up responses. As a result, DialoGPT can generate fluent and human-like dialogue without requiring task-specific supervision.

The **small** variant of DialoGPT is chosen for this project to balance performance and efficiency. It has a relatively compact parameter size compared to larger variants, making it suitable for rapid experimentation, fine-tuning, and reinforcement learning–based enhancement within limited computational resources.

Overall, DialoGPT-small provides a strong baseline conversational model that can be systematically improved while preserving response fluency and coherence.

---

##  Project Structure

```text
UmdTask_43_Fall2025_DialogueSystemEnhancement/
│
├── models/
│   ├── sft_finetuned/            # Output of supervised fine-tuning
│   ├── fine_tuned_live/          # Updated model during RLHF (PPO)
│
|__ project_utils
|   |── pipeline/                    # Code used by the runnable RLHF feedback system
│   |  ├── base_model.py             # Loads the base pretrained model
│   |  ├── ppo_model.py              # Loads PPO-trained model
│   |  ├── reward.py                 # Custom reward function (semantics, coherence…)
│   |  ├── evaluation.py             # Metrics (coherence, BLEU, ROUGE, BERTScore…)
│   |  ├── user_feedback.py          # Save and track user feedback
│   |  ├── run_pipeline.py           # Main RLHF + Gradio app
│   |
├── ├──training/                     # Scripts used during actual model training
│      ├── preprocess.py             # Data cleaning & template creation
│      ├── config.py                 # Hyperparameters and paths
│      ├── evaluate.py               # SFT + PPO evaluation suite
│      ├── reward_function.py        # Same reward model used in PPO
│      ├── supervised_finetuning.py              # Supervised fine-tuning
│      ├── test_ppo_setup.py         # PPO smoke test
│
├── docker-compose.yml            # Multi-container setup
├── Dockerfile                    # Image for training/inference
├── requirements.txt              # Dependencies
└── README.md
├── trl.example.md       # Walkthrough of the pipeline
├── trl.utils.py         # Utils used inside notebooks
├── trl.API.ipynb        # API demonstration notebook
├── trl.API.md           # API layer documentation
├── trl.example.ipynb  # Full training + evaluation demo


```

---





---

## Workflow Overview

### Fine-tuning in Google Colab
The model is fine-tuned in Google Colab to leverage free GPU resources, which significantly reduce training time compared to a typical laptop CPU. Colab also avoids local hardware and dependency constraints, improving reliability and reproducibility.

1. **Preprocessing**  
   Uses preprocessed context–response pairs from the dataset, where `first` serves as the dialogue context and `second` as the target response.

2. **Model Loading**  
   Loads the pre-trained `microsoft/DialoGPT-small` model using Hugging Face Transformers as the initial conversational policy.

3. **Supervised Fine-Tuning**  
   Performs supervised fine-tuning on the context–response pairs to adapt the pre-trained model to the dialogue distribution of the dataset and stabilize learning before reinforcement learning.

4. **Reward Modeling**  
   Uses a sentiment classifier (`cardiffnlp/twitter-roberta-base-sentiment`) to score generated responses and provide a scalar reward signal reflecting response quality.

5. **Fine-Tuning with PPO**  
   Applies Proximal Policy Optimization (PPO) using `trl.PPOTrainer`, where:
   - responses are generated by the model,
   - rewards are computed using the sentiment model,
   - the policy is updated to improve future responses.

6. **Saving the Model**  
   Saves the final fine-tuned model and uploads it to Hugging Face for reuse and evaluation.



## Deployment and Evaluation Workflow (Docker)

Inside the Docker container, no model training is performed. The container is used strictly for controlled inference, evaluation, and comparison of trained models.

### Step 1: Load Base Model
The system loads the original, untouched **`microsoft/DialoGPT-small`** model from Hugging Face to serve as a baseline reference.

- Purpose: Establish baseline conversational behavior
- Source: `microsoft/DialoGPT-small`

---

### Step 2: Load PPO Fine-Tuned Model
The PPO-enhanced dialogue model trained using RLHF in Google Colab is loaded from Hugging Face.

- Purpose: Evaluate the impact of reinforcement learning
- Source: `VenkataSivaRajesh/RLHF_Enhanced_DialoGPT`

---

### Step 3: Baseline Evaluation (Base Model)
The base DialoGPT model is evaluated using a held-out evaluation dataset.

**Metrics computed:**
- Coherence (semantic similarity)
- Sentiment score
- BLEU
- ROUGE-L
- BERTScore
- Distinct-2 diversity

These metrics establish baseline performance before applying PPO-based improvements.

---

### Step 4: Enhanced Model Evaluation (PPO Model)
The same evaluation pipeline and metrics are applied to the PPO fine-tuned model.

- Ensures a fair, controlled before/after comparison
- Quantifies the effect of RLHF optimization

---

### Step 5: Human Feedback Collection (Gradio Interface)(Bonus)
An interactive Gradio interface allows users to chat with the PPO-enhanced model.

**Functionality:**
- Users provide **Thumbs Up (+1)**, **Thumbs Down (-1)**, or **Skip (0)** feedback
- Feedback is logged to disk (`user_feedback.json`)
- Reward signals are computed for analysis and demonstration

This step demonstrates how real user feedback can be collected in RLHF systems.



---

### Step 6: Evaluation of SFT Baseline Model (Bonus)
A separately trained **Supervised Fine-Tuned (SFT)** model is also evaluated using the same metrics.

**Purpose:**
- Compare three models:
  1. Base model
  2. SFT-only model
  3. PPO (RLHF) model

This comparison highlights the added value of reinforcement learning beyond supervised fine-tuning.

---

### Outcome
This workflow provides:
- Controlled model comparison
- Quantitative evaluation using standard NLP metrics
- Qualitative evaluation through human feedback
- A realistic and modular RLHF deployment setup



---

## 🚀 Run Instructions

### 1. Clone the Repository
Before running any commands, clone the project repository and navigate to the project directory:

```bash
git clone https://github.com/gpsaggese-org/umd_classes.git
cd umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask_43_Fall2025_trl_Dialogue_System_Enhancement

```

### 2. Build Docker image
```bash
docker compose build 
```

### 3. Run container
```bash
docker compose up
```

### 4.Output
Once container runs:
 - Loads base model
 - Loads Fine tunde model
 - Runs Evaluation
 - Generates link to gradio

###  Open Gradio:
- Open the public Gradio URL displayed in the terminal
- Start interacting with the dialogue system through the web interface

---

##  Dependencies

```bash
transformers
trl
datasets
accelerate
scikit-learn
textblob
evaluate
torch
```

---

##  Credits

- **PPO implementation**: [TRL Library](https://github.com/huggingface/trl)
- **Sentiment model**: CardiffNLP RoBERTa-base
- **Dialogue dataset**: `pixelsandpointers/daily_dialog_w_turn_templates`
