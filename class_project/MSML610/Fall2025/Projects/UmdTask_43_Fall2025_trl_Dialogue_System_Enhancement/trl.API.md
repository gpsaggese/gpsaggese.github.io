# Dialogue System Enhancement — Internal API Documentation
*(For TRL-based DialogueRL Environment)*

---

## Overview

This document describes the **internal programming interface (API)** used in the DialogueRL system, which enhances a DialoGPT-based conversational model using reinforcement learning (TRL + PPO).  
It includes:

- Native low-level APIs  
- Reward model APIs  
- PPO training APIs  
- Wrapper layer APIs  
- Alternatives & comparisons  
- Recommendation  



## 🧠 Reinforcement Learning from Human Feedback (RLHF)
 
What it is (general):
RLHF is a training approach where a model learns not just from static datasets, but also from human preferences.
Instead of optimizing only likelihood (as in supervised fine-tuning), RLHF gives the model a reward signal based on what humans consider a good response. The model gradually improves by aligning its output with human intent, tone, and safety expectations.

How it is used in this project:
In this dialogue system, RLHF was applied after the initial initial ppo tuning.
Users interact with the model through a feedback interface (thumbs-up, thumbs-down, skip).
Each piece of feedback becomes a reinforcement signal:

- 👍 → positive reward

- 👎 → negative reward

- ⏭ → no update

The model uses this reward to refine its behavior in real time and learn conversational preferences directly from users.


## 🌀 Proximal Policy Optimization (PPO)

What it is (general):
PPO is a reinforcement learning algorithm widely used in language model alignment.
It updates the model's weights in small, controlled steps so that:

- the new responses improve according to the reward,

- the model doesn't drift too far from the base model,

- training remains stable and sample-efficient.

PPO is the standard for RLHF because it prevents the model from collapsing or overfitting after a few noisy feedback samples.

How it is used in this project:
This project uses TRL’s PPOTrainer to apply updates after each human interaction.

Workflow inside the system:

1.User sends a prompt → model generates a response.

2.User gives feedback → reward is calculated using my custom RewardFunction.

3.PPO updates the model with (prompt, response, reward) triplets.

4.The model is saved back to /ppo_dialogpt_model and improves incrementally.

This allows the model to learn conversational quality, coherence, and safety directly from the evaluator.





Training pipeline:

SFT Stage:
DialoGPT is fine-tuned on the pixelsandpointers/daily_dialog_w_turn_templates dataset to learn clean two-turn conversations.

RLHF Stage:
The SFT model is wrapped inside AutoModelForCausalLMWithValueHead so PPO can compute value estimates.
The Gradio interface allows human feedback to continuously adapt DialoGPT to:

- reduce echoing,

- avoid off-topic answers,

- generate safer responses,

- maintain context coherence.

In short, DialoGPT is the backbone, and PPO + RLHF provide the alignment layer on top.

# 1. Native Programming Interface (Core Internal APIs)

This section documents the raw internal functions, classes, and config objects used in the project.

---

## 1.1 Model Initialization API

### **load_base_model(model_name: str) → (tokenizer, model)**

Loads a pre-trained conversational model from HuggingFace.

**Parameters**
- `model_name` — e.g., `"microsoft/DialoGPT-small"`

**Returns**
- `tokenizer` — HuggingFace tokenizer  
- `model` — causal LM

**Notes**
- EOS token is used as pad token  
- Model automatically moved to CPU/GPU device  

---

## 1.2 Response Generation API

### **generate(model, tokenizer, prompt: str, max_new_tokens=60, history=None) → str**

Generates text using the language model.

**Parameters**
- `model`
- `tokenizer`
- `prompt`
- `max_new_tokens`
- `history` — conversation list for multi-turn chat

**Returns**
- Cleaned model reply (without echoing the prompt)

**Features**
- Supports one-shot & multi-turn dialogue  
- Removes DialoGPT prompt-attachment artifact  

---


## 1.3 Reward Model APIs

### **sentiment_score(text: str) → float**
- Uses `cardiffnlp/twitter-roberta-base-sentiment`
- Reward = positive − negative sentiment

### **toxicity_score(text: str) → float**
- Uses `unitary/unbiased-toxic-roberta`
- Reward = toxicity probability (penalty)

### **coherence_score(prompt: str, reply: str) → float**
- Uses SentenceTransformer (`all-MiniLM-L6-v2`)
- Reward = cosine similarity

### **compute_reward(prompt, reply) → float**
Weighted combination of:
- **sentiment**
- **coherence**
- **toxicity**



---

## 1.4 PPO Fine-Tuning API (TRL Core)

### **PPOTrainer(config, model, tokenizer, reward_fn)**

Handles:
- rollout generation  
- reward computation  
- PPO gradient update  
- KL penalty  
- Value function training  

---

### **trainer.generate(tokenized_prompt, max_new_tokens)**

Generates responses for rollouts.

---

### **trainer.step(queries, responses, rewards)**

Performs a PPO optimization step.

**Inputs**
- `queries` — tokenized prompts  
- `responses` — generated tokens  
- `rewards` — reward floats  

**Output**
- PPO statistics (`objective`, `kl`, `value_loss`, etc.)

---

## 1.5 Evaluation API

### **BLEU / ROUGE / BERTScore**
- **compute_bleu(pred, ref)**
- **compute_rouge(pred, ref)**
- **compute_bertscore(pred, ref)**



### **distinct_2(text) → float**
Measures diversity.

---

# 2. Wrapper Layer API (Lightweight Abstraction)

The wrapper simplifies the raw APIs into a usable interface for training and feedback.

---

## 2.1 Class: **DialogueRLModel**

### **Attributes**
- `model`
- `tokenizer`
- `ppo_trainer`
- `reward_fn`

---

## 2.2 Methods

### **generate_reply(prompt: str, history=None) → str**
User-facing text generation.

### **apply_feedback(prompt: str, reply: str, score: int) → dict**
Used for human-feedback-based PPO updates.

Process:
1. Tokenize prompt + reply  
2. Compute reward  
3. PPO update  
4. Return PPO logs  

### **save_model(path)**
Save trained PPO policy.

### **load_model(path)**
Load PPO policy.

---

## 3. Alternatives and Comparisons

### **Alternative 1: DialoGPT Without RL**

#### Advantages
- Lightweight  
- Easy to run locally  
- No PPO overhead  

#### Limitations
- Repeats prompt  
- No optimization  
- No safety controls  


### **Alternative 2: Supervised Fine-Tuning**

#### Advantages
- Simple training  
- Works well with labeled datasets  

#### Limitations
- Requires labeled data  
- Cannot optimize sentiment or safety directly  


### **Alternative 3: TRL PPO (Our Method)**

#### Advantages
- Learns from reward functions  
- Works with human feedback  
- Improves coherence, sentiment & toxicity  
- No labeled dataset required  

#### Limitations
- Higher memory usage  
- Requires careful tuning  


### **Alternative 4: Larger Chat Models (LLaMA, Mistral)**

#### Advantages
- Higher conversational quality  
- Context-rich replies  

#### Limitations
- Requires 16–48GB GPU  
- Not suitable for Docker/CPU setups  


---

## 4. Recommendation

For RL experimentation in constrained environments, DialoGPT + TRL PPO achieves the optimal trade-off between:

- customizability  
- low resource requirements  
- reward experimentation  
- measurable improvements  

This method is ideal when:

- you want RLHF-style fine-tuning  
- labeled data is limited  
- safety and coherence improvements matter  


## Our Integration Layer

### Goals

1. **Unified Dialogue Pipeline:** Provide a single interface to load the model, tokenizer, reward functions, and PPO trainer without exposing low-level TRL configuration.
2. **Modular Reward Design:** Allow sentiment, coherence, and toxicity rewards to be combined, extended, or replaced with minimal code changes.
3. **Seamless Human Feedback Loop:** Enable users to generate a reply, apply feedback, and update PPO weights in one consistent workflow.
4. **Simple Model Management:** One-line functions to save, load, and resume PPO-trained checkpoints for classroom and research use.
5. **Structured Outputs:** Ensure clean return types (reply, score logs, PPO stats) for integration with evaluation pipelines and UI tools (e.g., Gradio).

---




## Conclusion

This integration layer transforms TRL’s low-level PPO training flow into a streamlined, student-friendly interface suitable for dialogue enhancement tasks.  
By wrapping model loading, reward computation, feedback application, and PPO updates into a cohesive API, it simplifies experimentation and reduces overhead for educational environments.

The layer supports modular reward-function design, enables rapid testing with human feedback, and ensures model outputs remain structured and consistent.  
Overall, this integration provides a clear, extensible foundation for reinforcement-learning–based dialogue improvement projects such as DialoGPT + TRL PPO.

