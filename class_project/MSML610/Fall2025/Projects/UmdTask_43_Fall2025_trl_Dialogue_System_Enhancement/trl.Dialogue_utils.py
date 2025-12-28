import torch
from trl import PPOConfig
from datasets import load_dataset
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer
import evaluate
from typing import Dict, List, Any


def get_ppo_config() -> PPOConfig:
    """
    Return a PPOConfig that is:
    - CPU friendly
    - Small batch size to avoid OOM in WSL
    """
    cfg = PPOConfig()  # start from defaults so we don't fight version changes

    # Small, safe batch size – we will enforce this in main.py
    cfg.batch_size = 32
    cfg.mini_batch_size = 8

    # Conservative learning rate
    cfg.learning_rate = 1e-5

    # Make sure we stay on CPU
    if hasattr(cfg, "device"):
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"


    # Turn off fp16 / bf16 if those flags exist in this TRL version
    for attr in ["bf16", "fp16", "use_fp16", "use_bf16"]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, False)

    # Different TRL versions use different names for the PPO epoch count
    if hasattr(cfg, "num_ppo_epochs"):
        cfg.num_ppo_epochs = 1
    if hasattr(cfg, "ppo_epochs"):
        cfg.ppo_epochs = 1

    return cfg


# #############################################################################
# Preprocess API
# #################################################################################




def load_dataset_for_rl():
    """
    Load HF dataset and return a Python list of {"first": str, "second": str}
    Guaranteed row-by-row format.
    """
    raw = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates")
    train = raw["train"]

    pairs = []

    n = len(train["first"])
    for i in range(n):
        f = train["first"][i]
        s = train["second"][i]

        # Convert list → string
        if isinstance(f, list):
            f = " ".join(f)
        if isinstance(s, list):
            s = " ".join(s)

        # Clean
        if isinstance(f, str):
            f = f.strip()
        if isinstance(s, str):
            s = s.strip()

        if not isinstance(f, str) or not isinstance(s, str):
            continue
        if len(f) < 2 or len(s) < 2:
            continue

        pairs.append({"first": f, "second": s})

    print("Final cleaned pairs:", len(pairs))
    return pairs



# #############################################################################
# Reward Function API
# #############################################################################





class RewardFunction:
    def __init__(self, device=None):

        # Auto detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # -----------------------------------
        # 1. Sentiment Reward
        # -----------------------------------
        self.sent_tok = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        ).to(device)

        # -----------------------------------
        # 2. Coherence Reward (MiniLM)
        # -----------------------------------
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2", device=device
        )

        # -----------------------------------
        # Reward Weights 
        # -----------------------------------
        self.w_sent = 1.0
        self.w_coh  = 1.0
        self.w_len  = 0.3
        self.w_div  = 0.1
        self.w_rep  = 1.0

        # NEW: Human feedback weight
        self.w_hf   = 2.0      # HUMAN FEEDBACK IS IMPORTANT


    # =====================================================================
    # HELPER: Load human feedback from JSON
    # =====================================================================
    def _load_feedback(self):
        if not os.path.exists("user_feedback.json"):
            return []
        try:
            return json.load(open("user_feedback.json", "r"))
        except:
            return []


    # ====================================================================
    # HUMAN FEEDBACK REWARD
    # =====================================================================
    def _human_feedback_reward(self, prompts, responses):
        """
        Looks up (prompt, response) pairs in user_feedback.json and gives 
        +1 or -1 reward depending on stored feedback.
        """
        fb = self._load_feedback()
        rewards = []

        for p, r in zip(prompts, responses):

            matched = [
                entry["score"]
                for entry in fb
                if entry["prompt"] == p and entry["response"] == r
            ]

            if len(matched) == 0:
                rewards.append(0.0)
            else:
                rewards.append(sum(matched))

        return torch.tensor(rewards, device=self.device, dtype=torch.float32)


    # =====================================================================
    # Senstiment Feedback
    # =====================================================================
    @torch.no_grad()
    def _sentiment(self, texts):
        enc = self.sent_tok(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        logits = self.sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        neg, neu, pos = probs[:, 0], probs[:, 1], probs[:, 2]
        return pos - neg


    @torch.no_grad()
    def _coherence(self, prompts, responses):
        emb_p = self.embedder.encode(prompts, convert_to_tensor=True)
        emb_r = self.embedder.encode(responses, convert_to_tensor=True)
        sims = util.cos_sim(emb_p, emb_r).diagonal()
        return sims.clamp(0, 1)


    def _length_reward(self, responses, tokenizer):
        true_lengths = [len(tokenizer.encode(r, add_special_tokens=False))
                        for r in responses]

        lengths = torch.tensor(true_lengths, dtype=torch.float32, device=self.device)

        good = (lengths >= 6) & (lengths <= 30)

        return torch.where(
            good,
            torch.tensor(0.2, device=self.device),
            torch.tensor(-0.2, device=self.device)
        )


    def _distinct2(self, responses):
        vals = []
        for r in responses:
            toks = r.split()
            if len(toks) < 2:
                vals.append(0.0)
                continue
            bigrams = list(zip(toks, toks[1:]))
            vals.append(len(set(bigrams)) / len(bigrams))

        return torch.tensor(vals, device=self.device)


    def _repetition_penalty(self, prompts, responses):
        penalties = []
        for p, r in zip(prompts, responses):
            p_low = p.lower().strip()
            r_low = r.lower().strip()
            if r_low.startswith(p_low[:40]):
                penalties.append(-1.0)
            elif p_low in r_low[:len(p_low)]:
                penalties.append(-0.5)
            else:
                penalties.append(0.0)
        return torch.tensor(penalties, device=self.device)


    # =====================================================================
    # MASTER REWARD (NOW WITH HUMAN FEEDBACK)
    # =====================================================================
    @torch.no_grad()
    def __call__(self, prompts, responses, gen_tokenizer=None):

        assert gen_tokenizer is not None, \
            "Pass generation tokenizer to RewardFunction.__call__()"

        # Original components
        s_sent = self._sentiment(responses)
        s_coh  = self._coherence(prompts, responses)
        s_len  = self._length_reward(responses, gen_tokenizer)
        s_div  = self._distinct2(responses)
        s_rep  = self._repetition_penalty(prompts, responses)

        # NEW: HUMAN FEEDBACK component
        s_hfb = self._human_feedback_reward(prompts, responses)

        # Total reward
        final = (
            self.w_sent * s_sent +
            self.w_coh  * s_coh  +
            self.w_len  * s_len  +
            self.w_div  * s_div  -
            self.w_rep  * s_rep  +
            self.w_hf   * s_hfb
        )

        return final.cpu().tolist()
    



# #############################################################################
# test_ppo_setup API
# #############################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLMWithValueHead.from_pretrained("microsoft/DialoGPT-small")

print("Loading PPO config...")
ppo_config = get_ppo_config()

print("Building PPO trainer...")
trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer
)



print("\n🎉 PPO Trainer created successfully!")




# #############################################################################
# Main.py API file
# #############################################################################

import os
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig

from preprocess import load_dataset_for_rl
from reward import RewardFunction


# ----------------------------------------
# Safe tokenization
# ----------------------------------------
def safe_tokenize(text, tokenizer):
    try:
        tok = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=128,
        )
        ids = tok["input_ids"].squeeze(0)
        if ids.numel() == 0:
            return None
        return ids
    except:
        return None


# ============================================================
#                     MAIN TRAINING LOOP
# ============================================================
def main():

    print("\n=== Loading tokenizer & model ===")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

    # GPT2/DialoGPT has no pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "microsoft/DialoGPT-small"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ============================================================
    # Load your cleaned dataset
    # ============================================================
    print("\n=== Loading dataset via preprocess.py ===")
    dataset = load_dataset_for_rl()   # returns [{"first":..., "second":...}, ...]
    prompts = [row["first"] for row in dataset]

    print("Total prompts loaded:", len(prompts))

    # ============================================================
    # PPO CONFIG (optimized for 1000-step training)
    # ============================================================
    ppo_config = PPOConfig(
        batch_size=32,
        mini_batch_size=8,
        ppo_epochs=2,
        learning_rate=1.5e-6,    # more learning than 1e-6, still safe
        adap_kl_ctrl=True,
        init_kl_coef=0.15,
        target_kl=0.03,
        log_with=None,
    )

    print("\n=== Initializing PPO Trainer ===")
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )

    reward_fn = RewardFunction(device=device)

    # ============================================================
    # GENERATION SETTINGS
    # ============================================================
    gen_kwargs = {
        "max_new_tokens": 40,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "top_p": 0.90,
        "temperature": 0.8,
    }

    # ============================================================
    # START PPO TRAINING
    # ============================================================
    print("\n=== Starting PPO Training ===")

    MAX_STEPS = 1000  # ← your requirement

    for step in range(MAX_STEPS):

        start = step * ppo_config.batch_size
        end = (step + 1) * ppo_config.batch_size
        batch_prompts = prompts[start:end]

        if len(batch_prompts) == 0:
            print("\nReached end of dataset. Stopping early.")
            break

        print(f"\n---- PPO Step {step+1}/{MAX_STEPS} ----")
        print("Batch size:", len(batch_prompts))

        # --------------------------
        # TOKENIZATION
        # --------------------------
        query_tensors = []
        clean_prompts = []

        for p in batch_prompts:
            ids = safe_tokenize(p, tokenizer)
            if ids is not None:
                query_tensors.append(ids.to(device))
                clean_prompts.append(p)

        if len(query_tensors) == 0:
            print("SKIP → No valid prompts in this batch.")
            continue

        # --------------------------
        # GENERATION
        # --------------------------
        with torch.no_grad():
            response_tensors = trainer.generate(query_tensors, **gen_kwargs)

        decoded_responses = [
            tokenizer.decode(r, skip_special_tokens=True).strip()
            for r in response_tensors
        ]

        print("Sample response:", decoded_responses[0][:200])

        # --------------------------
        # REWARD
        # --------------------------
        rewards = reward_fn(clean_prompts, decoded_responses, tokenizer)

        reward_tensors = [
            torch.tensor(r, dtype=torch.float32, device=device)
            for r in rewards
        ]

        # --------------------------
        # PPO UPDATE
        # --------------------------
        trainer.step(query_tensors, response_tensors, reward_tensors)

        avg_reward = sum(rewards) / len(rewards)
        print("Avg reward this batch:", avg_reward)

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================
    out_dir = "ppo_dialogpt_model"
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\n=== Training complete. Model saved to: {out_dir} ===")


if __name__ == "__main__":
    main()


# ============================================================
# Bonus:- Feedback Loop
# ============================================================

import json
import torch
import gradio as gr
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig
from reward import RewardFunction
import os
import sys


# ---------------------------------------------------
# Save feedback to user_feedback.json
# ---------------------------------------------------
def save_feedback(prompt, response, score):
    try:
        fb = json.load(open("user_feedback.json", "r"))
    except:
        fb = []

    fb.append({
        "prompt": prompt,
        "response": response,
        "score": score
    })

    with open("user_feedback.json", "w") as f:
        json.dump(fb, f, indent=2)


# ---------------------------------------------------
# Load model + trainer one time
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "/content/ppo_dialogpt_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_DIR).to(device)

ppo_config = PPOConfig(
    batch_size=1,
    mini_batch_size=1,
    ppo_epochs=1,
    learning_rate=1e-6
)

trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer
)

reward_fn = RewardFunction(device=device)


# ---------------------------------------------------
# GENERATE MODEL RESPONSE
# ---------------------------------------------------
def generate_reply(prompt):
    if not prompt or prompt.strip() == "":
        return "Error: Empty input.", ""

    encoded = tokenizer(prompt, return_tensors="pt")
    query_tensor = encoded["input_ids"][0].to(device)

    response_tensors = trainer.generate(
        [query_tensor],
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    response = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
    return response, ""  # empty PPO log at generation time


# ---------------------------------------------------
# FEEDBACK HANDLER (with PPO stats)
# ---------------------------------------------------
def apply_feedback(prompt, score):
    global last_prompt, last_response

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build reward from HF score
    reward_val = float(score)
    reward_tensor = torch.tensor([reward_val], dtype=torch.float32).to(device)

    # Tokenize text → query tensor
    q = tokenizer(
        last_prompt,
        return_tensors="pt",
        truncation=True
    )["input_ids"].to(device)

    # Tokenize model reply → response tensor
    r = tokenizer(
        last_response,
        return_tensors="pt",
        truncation=True
    )["input_ids"].to(device)

    # PPO update (TRL 0.11.4 syntax)
    ppo_result = trainer.step(
        [q[0]],        # list of 1 query
        [r[0]],        # list of 1 response
        [reward_tensor]  # list of 1 reward
    )

    # Save model after update
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Format PPO logs for the UI
    formatted_stats = "\n".join([f"{k}: {v}" for k, v in ppo_result.items()])

    return f"Feedback applied (score={score}). PPO updated.\n\n{formatted_stats}"


def thumbs_up(prompt, response):
    return apply_feedback(prompt, response, 1)


def thumbs_down(prompt, response):
    return apply_feedback(prompt, response, -1)


def skip_feedback(prompt, response):
    return "Skipped — No training performed.", ""


# ---------------------------------------------------
# STOP PROGRAM (not only server)
# ---------------------------------------------------
def stop_everything():
    os._exit(0)


# ---------------------------------------------------
# GRADIO UI
# ---------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 PPO Dialogue Model — Human Feedback Interface")

    user_in = gr.Textbox(label="Your message:")
    model_out = gr.Textbox(label="Model reply:", interactive=False)
    ppo_logs = gr.Textbox(label="Status / PPO Logs", interactive=False)

    gen_btn = gr.Button("Generate Response")

    with gr.Row():
        up_btn = gr.Button("👍 Thumbs Up (+1)")
        down_btn = gr.Button("👎 Thumbs Down (-1)")
        skip_btn = gr.Button("⏭ Skip (0)")
        stop_btn = gr.Button("🔴 Stop Server & Program")

    # Generation
    gen_btn.click(generate_reply, inputs=user_in, outputs=[model_out, ppo_logs])

    # Feedback
    up_btn.click(thumbs_up, inputs=[user_in, model_out], outputs=[model_out, ppo_logs])
    down_btn.click(thumbs_down, inputs=[user_in, model_out], outputs=[model_out, ppo_logs])
    skip_btn.click(skip_feedback, inputs=[user_in, model_out], outputs=[model_out, ppo_logs])

    # Stop everything fully
    stop_btn.click(stop_everything, None, None)

demo.launch(share=True)


# ---------------------------------------------------
# evaluating base model
# ---------------------------------------------------

# ===========================
# 1. Imports
# ===========================
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================
# 2. Load Dataset (TEST SPLIT)
# ===========================
ds = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates", split="test")

N = 200   # evaluate on first 200 samples
prompts = ds["first"][:N]
ground_truth = ds["second"][:N]


# ===========================
# 3. Load FINE-TUNED PPO Model
# ===========================
model_path = "microsoft/DialoGPT-small"   # <-- YOUR FINE TUNED MODEL HERE
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

def generate_response(prompt):
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp,
        max_new_tokens=40,
        pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()

generated = [generate_response(p) for p in prompts]


# ===========================
# 4. Embedding Model (Coherence)
# ===========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def coherence_score(p, r):
    emb_p = embedder.encode(p, convert_to_tensor=True)
    emb_r = embedder.encode(r, convert_to_tensor=True)
    return util.cos_sim(emb_p, emb_r).item()

coherences = [coherence_score(p, g) for p, g in zip(prompts, generated)]
print("Coherence:", np.mean(coherences))


# ===========================
# 5. Sentiment Score
# ===========================
sent_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sent_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
).to(device)

def sentiment_score(text):
    enc = sent_tok(text, return_tensors="pt", truncation=True).to(device)
    logits = sent_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    neg, neu, pos = probs[0]
    return (pos - neg).item()

sentiments = [sentiment_score(g) for g in generated]
print("Sentiment:", np.mean(sentiments))





# ===========================
# 7. BLEU Score
# ===========================
bleu = evaluate.load("bleu")
bleu_results = bleu.compute(predictions=generated, references=[[gt] for gt in ground_truth])
print("BLEU:", bleu_results["bleu"])


# ===========================
# 8. ROUGE Score
# ===========================
rouge = evaluate.load("rouge")
rouge_results = rouge.compute(predictions=generated, references=ground_truth)
print("ROUGE-L:", rouge_results["rougeL"])


# ===========================
# 9. BERTScore
# ===========================
bertscore = evaluate.load("bertscore")
bertscore_results = bertscore.compute(
    predictions=generated, references=ground_truth, lang="en"
)
print("BERTScore (F1):", np.mean(bertscore_results["f1"]))


# ===========================
# 10. Diversity (Distinct-2)
# ===========================
def distinct_2(text):
    tokens = text.split()
    if len(tokens) < 2:
        return 0
    pairs = list(zip(tokens, tokens[1:]))
    return len(set(pairs)) / len(pairs)

distinct_scores = [distinct_2(g) for g in generated]
print("Distinct-2:", np.mean(distinct_scores))


# ===========================
# 11. Print some examples
# ===========================
print("\n================ EXAMPLES ================\n")
for i in range(10):
    print(f"Prompt:        {prompts[i]}")
    print(f"Fine-Tuned:    {generated[i]}")
    print("----------------------------------------\n")



# ==========================
# Evaluting Fine-tuned model
# ===========================

# ===========================
# 1. Imports
# ===========================



def post_evaluate(model_path: str, n_samples: int = 200) -> Dict[str, Any]:
    """
    Evaluates a fine-tuned causal language model (like DialoGPT) on key dialogue metrics.

    Args:
        model_path: Path to the fine-tuned Hugging Face model (e.g., "ppo_dialogpt_model").
        n_samples: Number of samples from the test set to evaluate.

    Returns:
        A dictionary containing the mean scores for all calculated metrics.
    """
    
    # ===========================
    # 1. Setup and Data Load
    # ===========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data (using the first turn as prompt, second as ground truth)
    ds = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates", split="test")
    prompts = ds["first"][:n_samples]
    ground_truth = ds["second"][:n_samples]
    print(f"Loaded {len(prompts)} samples for evaluation.")

    # Load FINE-TUNED PPO Model
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    
    # ===========================
    # 2. Response Generation
    # ===========================
    def generate_response(prompt: str) -> str:
        """Generates and cleans the model's response."""
        inp = tok(prompt, return_tensors="pt").to(device)

        out = model.generate(
            **inp,
            max_new_tokens=40,
            pad_token_id=tok.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.4
        )

        # CRITICAL: Remove the input prompt from the generated sequence
        gen_ids = out[0][inp["input_ids"].shape[1]:]

        reply = tok.decode(gen_ids, skip_special_tokens=True).strip()
        # Optional cleanup (remove leading punctuation)
        reply = reply.lstrip(" .,-:;")

        return reply

    generated: List[str] = []
    print("Generating responses...")
    for p in prompts:
        # Use a try-except to handle potential long-running or CUDA errors gracefully
        try:
            generated.append(generate_response(p))
        except Exception as e:
            print(f"Generation failed for a prompt: {e}. Appending empty string.")
            generated.append("")
    
    # Remove empty generations to avoid breaking metrics like BLEU/ROUGE
    valid_pairs = [(p, g, gt) for p, g, gt in zip(prompts, generated, ground_truth) if g]
    if not valid_pairs:
        print("No valid responses generated. Returning empty metrics.")
        return {}
        
    prompts, generated, ground_truth = zip(*valid_pairs)
    ground_truth_ref = [[gt] for gt in ground_truth] # Required format for BLEU
    
    print(f"Successfully generated {len(generated)} valid responses.")
    
    # ===========================
    # 3. Metric Calculation
    # ===========================
    metrics: Dict[str, float] = {}
    
    # --- 3.1. Coherence Score (Sentence Transformer) ---
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    @torch.no_grad()
    def coherence_score(p: str, r: str) -> float:
        emb_p = embedder.encode(p, convert_to_tensor=True)
        emb_r = embedder.encode(r, convert_to_tensor=True)
        return util.cos_sim(emb_p, emb_r).item()

    coherences = [coherence_score(p, g) for p, g in zip(prompts, generated)]
    metrics["coherence_mean"] = float(np.mean(coherences))
    
    # --- 3.2. Sentiment Score ---
    sent_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment").to(device)

    @torch.no_grad()
    def sentiment_score(text: str) -> float:
        enc = sent_tok(text, return_tensors="pt", truncation=True).to(device)
        logits = sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        neg, neu, pos = probs[0]
        return (pos - neg).item()

    sentiments = [sentiment_score(g) for g in generated]
    metrics["sentiment_mean"] = float(np.mean(sentiments))

    # --- 3.3. Toxicity Score ---
    tox_tok = AutoTokenizer.from_pretrained("unitary/unbiased-toxic-roberta")
    tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/unbiased-toxic-roberta").to(device)

    @torch.no_grad()
    def toxicity_score(text: str) -> float:
        enc = tox_tok(text, return_tensors="pt", truncation=True).to(device)
        logits = tox_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        # Toxicity score is typically the probability of the 'toxic' class (index 1 for this model)
        return probs[0][1].item() 

    toxicities = [toxicity_score(g) for g in generated]
    metrics["toxicity_mean"] = float(np.mean(toxicities))

    # --- 3.4. BLEU, ROUGE, BERTScore (Reference-Based Metrics) ---
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=generated, references=ground_truth_ref)
    metrics["BLEU"] = bleu_results["bleu"]

    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=generated, references=ground_truth)
    metrics["ROUGE-L"] = rouge_results["rougeL"]

    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(
        predictions=generated, references=ground_truth, lang="en", device=device
    )
    metrics["BERTScore_F1"] = float(np.mean(bertscore_results["f1"]))

    # --- 3.5. Diversity (Distinct-2) ---
    def distinct_2(text: str) -> float:
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        pairs = list(zip(tokens, tokens[1:]))
        return len(set(pairs)) / len(pairs)

    distinct_scores = [distinct_2(g) for g in generated]
    metrics["Distinct-2"] = float(np.mean(distinct_scores))
    
    # ===========================
    # 4. Print Results and Examples
    # ===========================
    print("\n" + "="*50)
    print("           DIALOGUE MODEL EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"| {k:<20} | {v:.4f} |")
    print("="*50)

    print("\n================ EXAMPLES ================\n")
    for i in range(min(10, len(prompts))):
        print(f"Prompt:          {prompts[i]}")
        print(f"Ground truth:    {ground_truth[i]}")
        print(f"Fine-Tuned Gen:  {generated[i]}")
        print("----------------------------------------\n")
        
    return metrics

