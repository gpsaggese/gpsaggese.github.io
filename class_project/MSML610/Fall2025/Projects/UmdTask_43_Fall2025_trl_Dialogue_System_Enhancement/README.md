# Dialogue System Enhancement with TRL PPO

This project fine-tunes a pre-trained conversational model using **reinforcement learning with human feedback (RLHF)** to enhance dialogue response quality. The optimization uses **Proximal Policy Optimization (PPO)** from the `trl` library and targets **user satisfaction** as defined by sentiment and fluency metrics.

---

##  Dataset

**Name:** `peandrew/dialy_dialogue_with_recoginized_concept_raw`  
**Size:** ~72,000 dialogues  
**Structure:**
- `dialog`: list of utterances (multi-turn dialogues)
- `act`, `emotion`, `mention`, `source`, `target`: metadata (not used in training)

We extract context–response pairs from the `dialog` field by slicing the last few utterances as context and predicting the next one.

---

##  Model

- **Base Model:** `microsoft/DialoGPT-small` from Hugging Face
- **Fine-Tuning:** PPO using `trl` (Transformer Reinforcement Learning)
- **Reward Signal:**  
  - `+1` for positive sentiment  
  - `0` for neutral  
  - `-1` for negative  
  - Optional: penalize very short or long replies

---

##  Project Structure

```text
UmdTask_43_Fall2025_DialogueSystemEnhancement/
│
├── deliverables/                 # Notebooks + documentation for submission
│   ├── trl.example.md       # Walkthrough of the pipeline
│   ├── trl.utils.py         # Utils used inside notebooks
│   ├── trl.API.ipynb        # API demonstration notebook
│   ├── trl.API.md           # API layer documentation
│   ├── trl.example.ipynb  # Full training + evaluation demo
│
├── models/
│   ├── sft_finetuned/            # Output of supervised fine-tuning
│   ├── fine_tuned_live/          # Updated model during RLHF (PPO)
│
├── pipeline/                     # Code used by the runnable RLHF feedback system
│   ├── base_model.py             # Loads the base pretrained model
│   ├── ppo_model.py              # Loads PPO-trained model
│   ├── reward.py                 # Custom reward function (toxicity, semantics, coherence…)
│   ├── evaluation.py             # Metrics (coherence, BLEU, ROUGE, BERTScore…)
│   ├── user_feedback.py          # Save and track user feedback
│   ├── run_pipeline.py           # Main RLHF + Gradio app
│
├── training/                     # Scripts used during actual model training
│   ├── preprocess.py             # Data cleaning & template creation
│   ├── config.py                 # Hyperparameters and paths
│   ├── evaluate.py               # SFT + PPO evaluation suite
│   ├── reward_function.py        # Same reward model used in PPO
│   ├── train_sft.py              # Supervised fine-tuning
│   ├── test_ppo_setup.py         # PPO smoke test
│
├── docker-compose.yml            # Multi-container setup
├── Dockerfile                    # Image for training/inference
├── requirements.txt              # Dependencies
└── README.md

```

---





---

##  Workflow Overview
# Fine-tuning the model in colab- 
- Due to less computational power of my laptop i finetuned my model using colab T4 GPU
1. **Preprocessing**  
   Extracts context–response pairs from multi-turn `dialog` sequences.

2. **Model Loading**  
   Loads `DialoGPT-small` using Hugging Face Transformers and prepares it with PPOConfig.

3. **Reward Modeling**  
   Uses a sentiment classifier (`cardiffnlp/twitter-roberta-base-sentiment`) to evaluate the quality of generated responses.

4. **Fine-Tuning with PPO**  
   - Rollouts from model
   - Compute rewards
   - PPO update step via `trl.PPOTrainer`

5. **Saving the model**
   - This fine tuned model is then uploaded to Huggingface


# Deployment in docker
Inside Docker, no training occurs.
You only load models + evaluate them using a controlled interface.

This is the workflow followed by the application inside the container:
## **1. Load Base Model **
The system loads the original, untouched **DialoGPT-small** model to serve as the baseline reference from HF.

- Source: `microsoft/DialoGPT-small`
- Purpose: Compare raw model behavior against PPO-tuned model

 **File:** `pipeline/base_model.py`

---

## **2. Load PPO Fine-Tuned Model (PPo Enhanced)**
Next, the pipeline loads your RLHF-trained PPO model hosted on HuggingFace:
This model has been fine-tuned using RLHF and PPO in Colab—T4.

- Source: 'VenkataSivaRajesh/Rlhf_Enhanced_DialoGpt'

## **3. Pre-Evaluation (Before PPO Model)**
The system evaluates the **base DialoGPT model** using multiple language quality metrics:

- **Coherence** (semantic similarity)
- **Sentiment score**
- **BLEU**
- **ROUGE-L**
- **BERTScore**
- **Distinct-2 diversity**

These metrics establish baseline performance before applying PPO improvements.


---

## **4. Post-Evaluation (Using PPO Model)**
The same metrics are computed again using the **PPO fine-tuned model**, producing a quantitative before/after comparison.

#  Bonus Components (Optional Extensions)

In addition to the main evaluation pipeline, the project includes two bonus modules that demonstrate interactive RLHF feedback collection and SFT model analysis.

---

## **5. Human Feedback Collection (Gradio Interface)**

After completing pre- and post-evaluation of the base and PPO models, the system launches an interactive **human-feedback interface** using Gradio.

### **Purpose**
- Allow humans to chat with the PPO-fine-tuned model
- Provide **Thumbs Up (+1)**, **Thumbs Down (–1)**, or **Skip (0)** feedback
- Each feedback triggers:
  - JSON logging (`user_feedback.json`)
  - Reward calculation via `RewardFunction`
  - A **PPO training step**
  - Model updates saved back to disk

### **Outcome**
This module shows how real users can shape model behavior after deployment—an essential component of RLHF workflows.


---

## **6. Evaluation of the SFT Baseline Model**

As part of the extended analysis, the project also evaluates the **Supervised Fine-Tuned (SFT) model** trained separately in Colab.

### **Purpose**
- Provide a third comparison point between:
  1. Base model  
  2. PPO fine-tuned model  
  3. SFT model  

- Metrics evaluated:
  - Coherence
  - Sentiment score
  - BLEU
  - ROUGE-L
  - BERTScore
  - Distinct-2 diversity

### **Outcome**
This step demonstrates how **supervised fine-tuning alone** compares against **RLHF-PPO fine-tuning** in terms of fluency, safety, and coherence.

 **File:** `pipeline/evaluation.py` (SFT evaluation function)

---

## **Bonus Summary**

| Component | Description | File |
|----------|-------------|------|
| **Human-feedback RLHF loop** | Real-time chat + reward + PPO update | `pipeline/user_feedback.py` |
| **SFT model evaluation** | Computes metrics on the SFT baseline | `pipeline/evaluation.py` |
| **Why bonus?** | These demonstrate extended RLHF capability beyond simple evaluation | Included in project for completeness |

---

Together, these bonus sections highlight how the system supports **interactive RLHF** and **multi-model comparison**, making the project a complete demonstration of a modern dialogue-system enhancement pipeline.



---

## 🚀 Run Instructions

### Local (VS Code, Jupyter, or Colab GPU)

Install dependencies:
```bash
pip install -r requirements.txt

# Build Docker image
docker compose build .

# Run container
docker compose up

## 🚀 Quick Start

### 🔓 Open Gradio:
- open the public URL and start interacting

---

## 📦 Dependencies

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

## 🧠 Credits

- **PPO implementation**: [TRL Library](https://github.com/huggingface/trl)
- **Sentiment model**: CardiffNLP RoBERTa-base
- **Dialogue dataset**: `peandrew/dialy_dialogue_with_recoginized_concept_raw`
