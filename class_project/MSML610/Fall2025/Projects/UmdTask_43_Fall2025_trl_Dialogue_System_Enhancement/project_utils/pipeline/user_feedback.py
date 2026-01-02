import os
import json
import torch
import gradio as gr
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig
from pipeline.reward import RewardFunction


# ================================================================
# Setup
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "../models/fine_tuned_live")
os.makedirs(MODEL_DIR, exist_ok=True)

FINETUNED_REPO = "VenkataSivaRajesh/Rlhf_Enhanced_DialoGpt"


# ================================================================
# Load model + tokenizer
# ================================================================
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_REPO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(FINETUNED_REPO).to(device)


# PPO config
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


# ================================================================
# Save feedback
# ================================================================
def save_feedback(prompt, response, score):
    try:
        fb = json.load(open("user_feedback.json", "r"))
    except:
        fb = []

    fb.append({"prompt": prompt, "response": response, "score": score})

    with open("user_feedback.json", "w") as f:
        json.dump(fb, f, indent=2)


# ================================================================
# Generate model reply
# ================================================================
def generate_reply(prompt):
    encoded = tokenizer(prompt, return_tensors="pt")
    q = encoded["input_ids"][0].to(device)

    out = trainer.generate(
        [q],
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )[0]

    return tokenizer.decode(out, skip_special_tokens=True)


# ================================================================
# Apply PPO update
# ================================================================
def apply_feedback(prompt, response, score):

    save_feedback(prompt, response, score)

    q = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
    r = tokenizer(response, return_tensors="pt")["input_ids"][0].to(device)

    reward_val = reward_fn([prompt], [response], tokenizer)[0]
    reward_tensor = torch.tensor([reward_val], dtype=torch.float32).to(device)

    trainer.step([q], [r], [reward_tensor])

    trainer.model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    return f"Feedback applied (score={score}). Reward={reward_val:.4f}. Updated model saved."


# ================================================================
# UI actions
# ================================================================
def thumbs_up(prompt, response):
    return apply_feedback(prompt, response, 1)

def thumbs_down(prompt, response):
    return apply_feedback(prompt, response, -1)

def skip_feedback(prompt, response):
    return "Feedback skipped."


# ================================================================
# Gradio Interface (Non-blocking)
# ================================================================
def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 PPO Dialogue Model — RLHF Feedback Interface")

        user_in = gr.Textbox(label="Your message:")
        model_out = gr.Textbox(label="Response:", interactive=False)

        btn_gen = gr.Button("Generate Response")

        with gr.Row():
            up = gr.Button("👍 Thumbs Up")
            down = gr.Button("👎 Thumbs Down")
            skip = gr.Button("⏭ Skip")

        btn_gen.click(generate_reply, user_in, model_out)
        up.click(thumbs_up, [user_in, model_out], model_out)
        down.click(thumbs_down, [user_in, model_out], model_out)
        skip.click(skip_feedback, [user_in, model_out], model_out)

    # 👇 This lets pipeline CONTINUE running
    demo.launch(share=True, prevent_thread_lock=True)


# ================================================================
# Function called by pipeline Step 5
# ================================================================
def generate_feedback(pre, post):
    
    print("\nLaunching Gradio interface... (non-blocking)\n")

    launch_gradio()

    print("\nGradio server running in background.\n")

  



# ================================================================
# Optionally run feedback standalone
# ================================================================
if __name__ == "__main__":
    launch_gradio()
