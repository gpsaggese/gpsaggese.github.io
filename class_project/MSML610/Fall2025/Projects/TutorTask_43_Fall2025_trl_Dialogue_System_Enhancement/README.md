# Dialogue System Enhancement with TRL PPO

This project fine-tunes a pre-trained conversational model using **reinforcement learning with human feedback (RLHF)** to enhance dialogue response quality. The optimization uses **Proximal Policy Optimization (PPO)** from the `trl` library and targets **user satisfaction** as defined by sentiment and fluency metrics.

---

## 📊 Dataset

**Name:** `peandrew/dialy_dialogue_with_recoginized_concept_raw`  
**Size:** ~13,000 dialogues  
**Structure:**
- `dialog`: list of utterances (multi-turn dialogues)
- `act`, `emotion`, `mention`, `source`, `target`: metadata (not used in training)

We extract context–response pairs from the `dialog` field by slicing the last few utterances as context and predicting the next one.

---

## 🧠 Model

- **Base Model:** `microsoft/DialoGPT-small` from Hugging Face
- **Fine-Tuning:** PPO using `trl` (Transformer Reinforcement Learning)
- **Reward Signal:**  
  - `+1` for positive sentiment  
  - `0` for neutral  
  - `-1` for negative  
  - Optional: penalize very short or long replies

---

## 🧩 Project Structure

```text
DialogueSystemEnhancement/
├── DialogueRL_utils.py         # All utility functions: dataset loading, reward, PPO, metrics
├── DialogueRL_API.ipynb        # Demo of core APIs (model, reward, generation)
├── DialogueRL_API.md           # Documentation of the utils and API layer
├── DialogueRL_example.ipynb    # Full training and evaluation pipeline
├── DialogueRL_example.md       # Walkthrough of the end-to-end pipeline
├── Dockerfile                  # Environment definition (for Colab/VS Code GPU)
└── README.md                   # Project summary and instructions

```

---





---

## 🔁 Workflow Overview

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

5. **Evaluation**  
   Pre- and post-fine-tuning metrics:
   - **Sentiment distribution**
   - **Response diversity (Distinct-n)**
   - **Dialogue length and fluency**

---

## 🚀 Run Instructions

### Local (VS Code, Jupyter, or Colab GPU)

Install dependencies:
```bash
pip install -r requirements.txt

# Build Docker image
docker build -t dialogue-ppo .

# Run Jupyter inside container
docker run -p 8888:8888 -v $PWD:/app dialogue-ppo jupyter notebook --ip=0.0.0.0 --allow-root

## 🚀 Quick Start

### 🔓 Open notebooks:
- `DialogueRL_API.ipynb` – quick demo  
- `DialogueRL_example.ipynb` – full pipeline

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

## 📈 Results

| Metric             | Before RL | After RL |
|--------------------|-----------|----------|
| Positive Sentiment | 34.2%     | 63.7%    |
| Distinct-2         | 0.82      | 0.91     |
| Fluency (avg len)  | 6.3       | 8.1      |

> **Note:** Results depend on training duration and reward function tuning.

---

## 🧠 Credits

- **PPO implementation**: [TRL Library](https://github.com/huggingface/trl)
- **Sentiment model**: CardiffNLP RoBERTa-base
- **Dialogue dataset**: `peandrew/dialy_dialogue_with_recoginized_concept_raw`
